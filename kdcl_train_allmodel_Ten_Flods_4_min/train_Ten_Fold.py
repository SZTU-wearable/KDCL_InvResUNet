import torch
from torch import nn
import sys
from Model import model_dict
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import optim
import os
import sys
# from transformers import AdamW
import torch.optim as optim
from datetime import datetime
import argparse
from torch.utils.data import Dataset
out_path = "output.txt"

class MyDataset(Dataset):

    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):
        return torch.Tensor(np.array(self.x_data[index])), torch.Tensor(
            np.array(self.y_data[index]))
        
class KD_Loss_fun_AIL(nn.Module):

    def __init__(self, alpha=0.3, temperature=1):
        super(KD_Loss_fun_AIL, self).__init__()
        self.alpha = alpha
        self.temperature = temperature

    def forward(self, Stu_pred, Tea_pred, targets):
        Stu_loss = torch.mean(torch.pow((Stu_pred - targets), 2))

        eT = torch.pow((Tea_pred - targets), 2)
        n = eT.max() - eT.min()
        sita = torch.sub(1, eT, alpha=(1 / n))

        Tea_loss = torch.mean(torch.pow((Stu_pred - Tea_pred), 2) * sita)

        loss = torch.add(torch.mul(Stu_loss, self.alpha),
                         torch.mul(Tea_loss, 1 - self.alpha))
        return loss
    
def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print_and_write_to_file(f"Folder created: {folder_path}")
    else:
        print_and_write_to_file(f"Folder already exists: {folder_path}")
        
def print_and_write_to_file(text, file_path=out_path):
    # Printing to console
    print(text)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    text_with_timestamp = f"{timestamp}: {text}"
    
    # Writing to file
    with open(file_path, 'a') as file:
        file.write(text_with_timestamp + '\n')
        
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
  
        
def train_one_epoch(models, optimizers, train_loader, loss_fun, mae_fun, train_loader_tqdm):
    acc_recorder_list = []
    loss_recorder_list = []
    for model in models:
        model.train()
        acc_recorder_list.append(AverageMeter())
        loss_recorder_list.append(AverageMeter())
    
    for i, (inputs, targets) in enumerate(train_loader_tqdm):
        # torch.Size([batch, num_model, 3, 32, 32]) torch.Size([batch])
        outputs = torch.zeros(size=(len(models), inputs.size(0), 1, 625), dtype=torch.float).to(device)
        out_list = []
        # forward
        minn_val = float('inf')
        best_idx = -1
        for model_idx, model in enumerate(models):
            inputs = inputs.to(device)
            targets = targets.to(device)
            out = model.forward(inputs)
            
            acc = mae_fun(out, targets)
            if minn_val > acc.item():
                best_idx = model_idx
                minn_val = acc.item()
                
            outputs[model_idx, ...] = out
            out_list.append(out)

        # backward
        stable_out = outputs[best_idx]
        stable_out = stable_out.detach()

        for model_idx, model in enumerate(models):
            loss = loss_fun(out_list[model_idx], stable_out, targets)
            optimizers[model_idx].zero_grad()
            
            if model_idx < len(models) - 1:
                loss.backward(retain_graph=True)
            else:
                loss.backward()

            optimizers[model_idx].step()

            loss_recorder_list[model_idx].update(loss.item(), n=targets.size(0))
            acc = mae_fun(out_list[model_idx], targets)
            acc_recorder_list[model_idx].update(acc.item(), n=inputs.size(0))
        train_loader_tqdm.set_postfix(loss=f"{[recorder.avg for recorder in acc_recorder_list]}")
    losses = [recorder.avg for recorder in loss_recorder_list]
    acces = [recorder.avg for recorder in acc_recorder_list]
    return losses, acces

def evaluation(models, val_loader, loss_fun, mae_fun):
    acc_recorder_list = []
    loss_recorder_list = []
    for model in models:
        model.eval()
        acc_recorder_list.append(AverageMeter())
        loss_recorder_list.append(AverageMeter())

    with torch.no_grad():
        for inputs, targets in val_loader:
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                targets = targets.cuda()

            for model_idx, model in enumerate(models):
                out = model(inputs)
                acc = mae_fun(out, targets)
                loss = mae_fun(out, targets)
                acc_recorder_list[model_idx].update(acc.item(), inputs.size(0))
                loss_recorder_list[model_idx].update(loss.item(), inputs.size(0))
    losses = [recorder.avg for recorder in loss_recorder_list]
    acces = [recorder.avg for recorder in acc_recorder_list]
    return losses, acces


def train(model_list, optimizer_list, train_loader, val_loader, test_loader, model_names, loss_fun, mae_fun, save_path=None, patience=10, fold_number="00"):
    num_epochs = 100
    no_improvement = 0
    best_acc = [float('inf') for _ in range(len(model_list))]
    for epoch in range(num_epochs):
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        train_losses, train_acces = train_one_epoch(model_list, optimizer_list, train_loader, loss_fun, mae_fun, train_loader_tqdm)
        val_losses, val_acces = evaluation(model_list, val_loader, loss_fun, mae_fun)

        for i in range(len(best_acc)):
            if val_acces[i] < best_acc[i]:
                current_datetime = datetime.now()
                current_datetime_str = current_datetime.strftime('_%Y_%m_%d_%H:%M:%S')
                test_loss, test_mae = evaluation(model_list, test_loader, loss_fun, mae_fun)
                path_name = model_names[i] +"Fold"+ fold_number +current_datetime_str + "_testMAE_" + str(test_mae) +".pth"
                best_acc[i] = val_acces[i]
                state_dict = dict(epoch=epoch + 1, model=model_list[i].state_dict(),
                                  acc=val_acces[i])
                file_name = os.path.join(save_path, model_names[i], path_name)
                os.makedirs(os.path.dirname(file_name), exist_ok=True)
                torch.save(state_dict, file_name)
                
                if model_names[i] == "InvertedResidual_UNet_small":
                    no_improvement = 0
                
            elif model_names[i] == "InvertedResidual_UNet_small":
                no_improvement += 1
                print_and_write_to_file(
                    f'EarlyStopping counter: {no_improvement} out of {patience}'
                )
        print_and_write_to_file(best_acc)
                
        if no_improvement >= patience:
            print_and_write_to_file("Early Stopping! Training stopped.")
            break

        for j in range(len(best_acc)):
            print_and_write_to_file("model:{} train loss:{:.2f} acc:{:.2f}  val loss{:.2f} acc:{:.2f}".format(
                model_names[j], train_losses[j], train_acces[j], val_losses[j],
                val_acces[j]))

    for k in range(len(best_acc)):
        print_and_write_to_file("model:{} best acc:{:.2f}".format(model_names[k], best_acc[k]))




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 添加命令行参数
    parser.add_argument('-fold_num', type=str)
    parser.add_argument('-batch_size', type=int, default=256)
    parser.add_argument('-cuda_num', type=str, default='0')
    parser.add_argument('-save_path', type=str, default='weight_model/')

    # 解析命令行参数
    args = parser.parse_args()

    # 获取参数值
    fold_number = args.fold_num
    BATCH_SIZE = args.batch_size
    cuda_number = args.cuda_num
    weight_dir = args.save_path
    
    # with open(out_path, 'w') as file:
    #     file.write('START\n')
    
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_number
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_and_write_to_file(device)
    print_and_write_to_file(fold_number)
    # print(model_dict)
    model_list = []
    optimizer_list = []
    lr = 1e-3
    # del model_dict['UTransBPNet']
    model_names = list(model_dict.keys())
    print_and_write_to_file(model_names)
    for model_name, _ in model_dict.items():
        # print(model_name, model)
        model = model_dict[model_name]().to(device)
        model_list.append(model)
        # print(model)
        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=0.1)
        optimizer_list.append(opt)
        

    dataset_dir_path = '/home/yonghu/data_yonghu/code_python/VitalDB_DataLoader_tol/Folder' + fold_number + '/'

    train_data_set = torch.load(dataset_dir_path+"train_dataset.pt")
    val_data_set = torch.load(dataset_dir_path+"val_dataset.pt")
    test_data_set = torch.load(dataset_dir_path+"test_dataset.pt")

    print_and_write_to_file(len(train_data_set))
    print_and_write_to_file(len(val_data_set))
    print_and_write_to_file(len(test_data_set))


    train_loader = DataLoader(train_data_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data_set, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data_set, batch_size=BATCH_SIZE, shuffle=False)
    print_and_write_to_file([len(train_loader), len(val_loader), len(test_loader)])
    alphs = 0.2
    loss_fun = KD_Loss_fun_AIL(alphs)
    mae_fun = nn.L1Loss()
    patience = 10
    
    train(model_list, optimizer_list, train_loader, val_loader, test_loader, model_names, loss_fun, mae_fun, weight_dir, patience, fold_number)
