import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import (CosineAnnealingLR,
                                      CosineAnnealingWarmRestarts, StepLR,
                                      ExponentialLR)
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):

    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(hidden_dim,
                                           dim), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


# class Residual(nn.Module):
#     def __init__(self, fn):
#         super().__init__()
#         self.fn = fn
#     def forward(self, x, **kwargs):
#         return self.fn(x, **kwargs) + x


class Attention(nn.Module):

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x, mask=None):
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # 对tensor张量分块
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        # dim=128, depth=12, heads=8,
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    PreNorm(
                        dim,
                        Attention(dim,
                                  heads=heads,
                                  dim_head=dim_head,
                                  dropout=dropout)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask) + x
            x = ff(x) + x
        return x


class BPT(nn.Module):  # Transformer Encoder

    def __init__(self,
                 point_dim,
                 length,
                 kernel_size,
                 dim,
                 num_classes,
                 depth,
                 heads,
                 mlp_dim,
                 pool='cls',
                 dim_head=64,
                 dropout=0.,
                 emb_dropout=0.):
        super().__init__()
        assert pool in {
            'cls', 'mean'
        }, 'pool type must be either cls (cls token) or mean(mean pooling)'
        self.to_point_embedding = nn.Sequential(nn.Linear(point_dim, dim))
        # self.conv = ConvNormPool(point_dim,length,kernel_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, length + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim,
                                       dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim),
                                      nn.Linear(dim, num_classes))

    def forward(self, input, mask=None):
        x = self.to_point_embedding(input)  # x.shape = [b, 625, 128]
        # print("x!!!!:",x.shape)
        # x = self.conv(input)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)  # x.shape = [b, 626, 128]
        x += self.pos_embedding[:, :(n + 1)]  # x.shape = [b, 626, 128]
        x = self.dropout(x)

        x = self.transformer(x, mask)  # x.shape = [b, 626, 128]

        # x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        # print("x:",x.shape)
        # x = torch.cat((x,torch.transpose(input,1,2)[:,0,625:].squeeze(1)),1)

        return self.mlp_head(x[:, 1:, :])


class conbr_block(nn.Module):

    def __init__(self, in_layer, out_layer, kernel_size, stride, dilation):
        super(conbr_block, self).__init__()

        self.conv1 = nn.Conv1d(in_layer,
                               out_layer,
                               kernel_size=kernel_size,
                               stride=stride,
                               dilation=dilation,
                               padding=3,
                               bias=True)
        self.bn = nn.BatchNorm1d(out_layer)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        out = self.relu(x)

        return out


class se_block(nn.Module):

    def __init__(self, in_layer, out_layer):
        super(se_block, self).__init__()

        self.conv1 = nn.Conv1d(in_layer,
                               out_layer // 8,
                               kernel_size=1,
                               padding=0)
        self.conv2 = nn.Conv1d(out_layer // 8,
                               in_layer,
                               kernel_size=1,
                               padding=0)
        self.fc = nn.Linear(1, out_layer // 8)
        self.fc2 = nn.Linear(out_layer // 8, out_layer)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x_se = nn.functional.adaptive_avg_pool1d(x, 1)
        x_se = self.conv1(x_se)
        x_se = self.relu(x_se)
        x_se = self.conv2(x_se)
        x_se = self.sigmoid(x_se)

        x_out = torch.add(x, x_se)
        return x_out


class re_block(nn.Module):

    def __init__(self, in_layer, out_layer, kernel_size, dilation):
        super(re_block, self).__init__()

        self.cbr1 = conbr_block(in_layer, out_layer, kernel_size, 1, dilation)
        self.cbr2 = conbr_block(out_layer, out_layer, kernel_size, 1, dilation)
        self.seblock = se_block(out_layer, out_layer)

    def forward(self, x):

        x_re = self.cbr1(x)
        x_re = self.cbr2(x_re)
        x_re = self.seblock(x_re)
        x_out = torch.add(x, x_re)
        return x_out


class MultiHeadAttention(nn.Module):

    def __init__(self):
        super(MultiHeadAttention, self).__init__()

    def positional_encoding_2d(self, d_model, height, width):
        """
        reference: wzlxjtu/PositionalEncoding2D
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        try:
            pe = pe.to(torch.device("cuda:0"))
        except RuntimeError:
            pass
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(
            torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)
        return pe

    def forward(self, x):
        raise NotImplementedError()


import math


class MultiHeadDense(nn.Module):

    def __init__(self, d, bias=False):
        super(MultiHeadDense, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(d, d))
        if bias:
            raise NotImplementedError()
            self.bias = Parameter(torch.Tensor(d, d))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # x:[b, h*w, d]
        b, wh, d = x.size()
        x = torch.bmm(x, self.weight.repeat(b, 1, 1))
        # x = F.linear(x, self.weight, self.bias)
        return x


class PositionalEncoding1D(nn.Module):

    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding1D, self).__init__()
        channels = int(channels)
        self.channels = channels
        inv_freq = 1. / (10000
                         **(torch.arange(0, channels, 2).float() / channels))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 4d!")
        # print("tensor:",tensor.shape)
        batch_size, x, orig_ch = tensor.shape
        pos_x = torch.arange(x,
                             device=tensor.device).type(self.inv_freq.type())
        # pos_y = torch.arange(y,
        #                      device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        # sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
        # emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)
        emb = torch.zeros((x, self.channels),
                          device=tensor.device).type(tensor.type())

        emb[:, :self.channels] = emb_x

        return emb[None, :, :orig_ch].repeat(batch_size, 1, 1)


class PositionalEncodingPermute1D(nn.Module):

    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x) instead of (batchsize, x, ch)        
        """
        super(PositionalEncodingPermute1D, self).__init__()

        self.penc = PositionalEncoding1D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 1)
        enc = self.penc(tensor)
        # print("enc:",enc.shape)
        return enc.permute(0, 2, 1)


class MultiHeadCrossAttention(MultiHeadAttention):

    def __init__(self, channelY, channelS):
        super(MultiHeadCrossAttention, self).__init__()

        self.Sconv = nn.Sequential(
            nn.MaxPool1d(5), nn.Conv1d(channelS, channelS, kernel_size=1),
            nn.BatchNorm1d(channelS), nn.ReLU(inplace=True))

        self.Yconv = nn.Sequential(
            # nn.MaxPool1d(1),
            nn.Conv1d(channelY, channelS, kernel_size=1),
            nn.BatchNorm1d(channelS),
            nn.ReLU(inplace=True))
        self.query = MultiHeadDense(channelS, bias=False)
        self.key = MultiHeadDense(channelS, bias=False)
        self.value = MultiHeadDense(channelS, bias=False)

        self.conv = nn.Sequential(nn.Conv1d(channelS, channelS, kernel_size=1),
                                  nn.BatchNorm1d(channelS),
                                  nn.ReLU(inplace=True),
                                  nn.Upsample(scale_factor=5, mode='nearest'))

        self.Yconv2 = nn.Sequential(
            nn.Upsample(scale_factor=5, mode='nearest'),
            nn.Conv1d(channelY, channelY, kernel_size=3, padding=1),
            nn.Conv1d(channelY, channelS, kernel_size=1),
            nn.BatchNorm1d(channelS), nn.ReLU(inplace=True))

        self.softmax = nn.Softmax(dim=1)
        self.Spe = PositionalEncodingPermute1D(channelS)
        self.Ype = PositionalEncodingPermute1D(channelY)

    def forward(self, Y, S):
        Sb, Sc, Sh = S.size()
        Yb, Yc, Yh = Y.size()
        # print("S:",S.shape)
        Spe = self.Spe(S)
        S = S + Spe
        S1 = self.Sconv(S).reshape(Yb, Sc, Yh).permute(0, 2, 1)
        # print("S1:",S1.shape)
        V = self.value(S1)
        # print("V:",V.shape)
        Ype = self.Ype(Y)
        Y = Y + Ype
        Y1 = self.Yconv(Y).reshape(Yb, Sc, Yh).permute(0, 2, 1)
        # print("Y1:",Y1.shape)
        Y2 = self.Yconv2(Y)
        # print("Y2:",Y2.shape)
        Q = self.query(Y1)
        # print("Q:",Q.shape)
        K = self.key(Y1)
        # print("K:",K.shape)
        A = self.softmax(torch.bmm(Q, K.permute(0, 2, 1)) / math.sqrt(Sc))
        x = torch.bmm(A, V).permute(0, 2, 1).reshape(Yb, Sc, Yh)
        # print("after softmax x:",x.shape)
        Z = self.conv(x)
        # print("Z:",Z.shape)
        Z = Z * S
        # print("[Z.shape,Y2.shape]:",(Z.shape,Y2.shape))
        Z = torch.cat([Z, Y2], dim=1)
        return Z


class MultiHeadSelfAttention(MultiHeadAttention):

    def __init__(self, channel):
        super(MultiHeadSelfAttention, self).__init__()
        self.query = MultiHeadDense(channel, bias=False)
        self.key = MultiHeadDense(channel, bias=False)
        self.value = MultiHeadDense(channel, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.pe = PositionalEncodingPermute1D(channel)

    def forward(self, x):
        b, c, h = x.size()
        # pe = self.positional_encoding_2d(c, h, w)
        # print("X:",x.shape)
        pe = self.pe(x)
        # print("pe:",pe.shape)
        # pe: torch.Size([1, 78, 512])
        # x: torch.Size([1, 512, 78])
        # pe: torch.Size([1, 78, 256])
        # print("x:",x.shape)
        # x: torch.Size([1, 512, 78])
        x = x + pe
        x = x.reshape(b, c, h).permute(0, 2, 1)  #[b, h*w, d]
        Q = self.query(x)
        K = self.key(x)
        A = self.softmax(torch.bmm(Q, K.permute(0, 2, 1)) /
                         math.sqrt(c))  #[b, h*w, h*w]
        V = self.value(x)
        x = torch.bmm(A, V).permute(0, 2, 1).reshape(b, c, h)
        return x


class TransformerUp(nn.Module):

    def __init__(self, Ychannels, Schannels):
        super(TransformerUp, self).__init__()
        self.MHCA = MultiHeadCrossAttention(Ychannels, Schannels)
        # self.conv = nn.Conv1d(768,896,kernel_size=3,padding=1)

    def forward(self, Y, S):
        x = self.MHCA(Y, S)
        # print("foward_x:",x.shape)# 32 768 25
        # x = self.conv(x)
        return x


class Teacher_Model(nn.Module):

    def __init__(self, input_dim=4, layer_n=128, kernel_size=7, depth=3):
        super(Teacher_Model, self).__init__()
        self.input_dim = input_dim
        self.layer_n = layer_n
        self.kernel_size = kernel_size
        self.depth = depth

        self.AvgPool1D1 = nn.AvgPool1d(input_dim, stride=5)
        self.AvgPool1D2 = nn.AvgPool1d(input_dim, stride=25)
        self.AvgPool1D3 = nn.AvgPool1d(input_dim, stride=125)

        self.layer1 = self.down_layer(self.input_dim, self.layer_n,
                                      self.kernel_size, 1, 2)
        self.layer2 = self.down_layer(self.layer_n, int(self.layer_n * 2),
                                      self.kernel_size, 5, 2)
        self.layer3 = self.down_layer(
            int(self.layer_n * 2) + int(self.input_dim), int(self.layer_n * 3),
            self.kernel_size, 5, 2)
        self.layer4 = self.down_layer(
            int(self.layer_n * 3) + int(self.input_dim), int(self.layer_n * 4),
            self.kernel_size, 5, 2)
        # self.layer5 = self.down_layer(int(self.layer_n*4)+int(self.input_dim), int(self.layer_n*5), self.kernel_size,4, 2)

        self.vit = BPT(point_dim=5,
                       length=512,
                       kernel_size=3,
                       dim=256,
                       num_classes=5,
                       depth=12,
                       heads=8,
                       mlp_dim=512,
                       pool='mean')
        # self.MHSA = MultiHeadSelfAttention(512)

        # self.MHCA = MultiHeadCrossAttention(Ychannels, Schannels)
        self.up1 = TransformerUp(512, 384)
        self.up2 = TransformerUp(384, 256)
        self.up3 = TransformerUp(256, 128)

        # self.cbr_up1 = conbr_block(int(self.layer_n*7), int(self.layer_n*3), self.kernel_size, 1, 1)
        # self.cbr_up2 = conbr_block(int(self.layer_n*5), int(self.layer_n*2), self.kernel_size, 1, 1)
        # self.cbr_up3 = conbr_block(int(self.layer_n*3), self.layer_n, self.kernel_size, 1, 1)

        self.cbr_up1 = conbr_block(int(self.layer_n * 9),
                                   int(self.layer_n * 3), self.kernel_size, 1,
                                   1)
        self.cbr_up2 = conbr_block(int(self.layer_n * 6),
                                   int(self.layer_n * 2), self.kernel_size, 1,
                                   1)
        self.cbr_up3 = conbr_block(int(self.layer_n * 3), self.layer_n,
                                   self.kernel_size, 1, 1)

        self.upsample = nn.Upsample(scale_factor=5, mode='nearest')
        self.upsample1 = nn.Upsample(scale_factor=5, mode='nearest')

        self.outcov = nn.Conv1d(self.layer_n,
                                1,
                                kernel_size=self.kernel_size,
                                stride=1,
                                padding=3)

    def down_layer(self, input_layer, out_layer, kernel, stride, depth):
        block = []
        block.append(conbr_block(input_layer, out_layer, kernel, stride, 1))
        for i in range(depth):
            block.append(re_block(out_layer, out_layer, kernel, 1))
        return nn.Sequential(*block)

    def forward(self, x):

        pool_x1 = self.AvgPool1D1(x)
        # print("pool_x1:",pool_x1.shape)
        pool_x2 = self.AvgPool1D2(x)
        # print("pool_x2:",pool_x2.shape)
        pool_x3 = self.AvgPool1D3(x)
        # print("pool_x3:",pool_x3.shape)

        #############Encoder#####################
        # print("x:",x.shape)
        out_0 = self.layer1(x)
        # print("out_0.shape:",out_0.shape)
        out_1 = self.layer2(out_0)
        # print("out_1.shape:",out_1.shape)

        x = torch.cat([out_1, pool_x1], 1)
        # print("cat_1:",x.shape)
        out_2 = self.layer3(x)  # out_2.shape: torch.Size([32, 384, 25])
        # print("out_2.shape:",out_2.shape)
        x = torch.cat([out_2, pool_x2], 1)
        # print("cat_2:",x.shape)
        out_3 = self.layer4(x)
        # print("out_3.shape:", out_3.shape) # torch.Size([32, 512, 5])
        x = self.vit(out_3)  # vit: torch.Size([32, 512, 5])
        # print("x.shape:", x.shape) # torch.Size([32, 512, 5])
        #############Decoder#####################
        # x = self.upsample(x) # torch.Size([32, 512, 25])
        up = self.up1(x, out_2)
        up = torch.cat([up, out_2], dim=1)
        # print("up_1:", up.shape)
        up = self.cbr_up1(up)
        # print("up:",up.shape)
        # up = self.upsample(up)
        # print("up!!:",up.shape)
        up = self.up2(up, out_1)
        up = torch.cat([up, out_1], dim=1)
        # print("up_2:",up.shape)
        # print("up_2:",u p.shape)
        up = self.cbr_up2(up)
        # up = self.upsample(up)
        up = self.up3(up, out_0)
        up = torch.cat([up, out_0], dim=1)
        # print("up_3:",up.shape)
        up = self.cbr_up3(up)
        # print("up:",up.shape)
        out = self.outcov(up)
        return out

if __name__ == "__main__":
    model = Teacher_Model()
    
    from thop import profile
    from pytorch_model_summary import summary

    a = torch.randn(1, 4, 625)
    # print(model)
    # print(model.features)
    print("input shape: ", a.shape)
    print("output shape: ", model(a).shape)
    # summary(model, (4,625), device="cpu")
    summary(model,
            a,
            show_input=False,
            show_hierarchical=False,
            print_summary=True)
    flops, params = profile(model, inputs=(a,))
    print(f"模型的计算量估计：{flops / 1e9} GFLOPs")
    print(f"模型的参数数量：{params / 1e6} Million")
    # print_model_parameters(model)

