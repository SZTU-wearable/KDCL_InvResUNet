import torch
from torch import nn
import sys
sys.path.append('../Model')
from U_Net_1D import UNet
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


class MyDataset(Dataset):

    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):
        return torch.Tensor(np.array(self.x_data[index])), torch.Tensor(
            np.array(self.y_data[index]))


def train_model(model, train_loader, val_loader, optimizer, criterion=nn.L1Loss(), num_epochs=100, patience=7,
                save_path=None):
    best_val_loss = float('inf')
    no_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for id, (inputs, targets) in enumerate(train_loader_tqdm):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            train_loader_tqdm.set_postfix(loss=f"{total_loss / (id + 1):.4f}")
        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        with torch.no_grad():
            avg_val_loss = test_model(model, val_loader)
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Validation Loss: {avg_val_loss:.6f}, Best Loss: {best_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            print(
                f'Validation loss decreased ({best_val_loss:.6f} --> {avg_val_loss:.6f}).  Saving model ...'
            )
            best_val_loss = avg_val_loss
            no_improvement = 0

            if save_path is not None:
                current_datetime = datetime.now()
                current_datetime_str = current_datetime.strftime('_%Y_%m_%d_%H:%M:%S')
                test_mae = test_model(model, test_loader)
                path_name = save_path + current_datetime_str + "_testMAE_" + str(test_mae) +".pth"
                torch.save(model.state_dict(), path_name)
                print(f"Saved the best model to '{path_name}'.")
                global best_model
                best_model = model
            else:
                print("Not Save!!!!")

        else:
            no_improvement += 1
            print(
                f'EarlyStopping counter: {no_improvement} out of {patience}'
            )

        if no_improvement >= patience:
            print("Early Stopping! Training stopped.")
            break

    print("Training completed!")


# 测试模型
def test_model(model, test_loader):
    model.eval()
    true_data = []
    pred_data = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            true_data.append(targets.detach().cpu().numpy().flatten())
            pred_data.append(outputs.detach().cpu().numpy().flatten())
    true_data = np.concatenate(true_data)
    pred_data = np.concatenate(pred_data)
    mae = np.mean(np.abs(true_data - pred_data))
    print(f"Test Loss: {mae}")
    return mae


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 添加命令行参数
    parser.add_argument('-model_name', type=str)
    parser.add_argument('-fold_num', type=str)
    parser.add_argument('-batch_size', type=int, default=256)
    parser.add_argument('-cuda_num', type=str, default='0')

    # 解析命令行参数
    args = parser.parse_args()

    # 获取参数值
    MODEL_NAME = args.model_name
    fold_number = args.fold_num
    BATCH_SIZE = args.batch_size
    cuda_number = args.cuda_num
    
    
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_number
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = UNet(4, 1).to(device)
    dataset_dir_path = '/home/yonghu/data_yonghu/code_python/VitalDB_DataLoader_tol/Folder' + fold_number + '/'
    weight_path = 'weight_model/'+MODEL_NAME+'_VitalDB_Fold'+ fold_number +'_'
    train_data_set = torch.load(dataset_dir_path+"train_dataset.pt")
    val_data_set = torch.load(dataset_dir_path+"val_dataset.pt")
    test_data_set = torch.load(dataset_dir_path+"test_dataset.pt")

    print(len(train_data_set))
    print(len(val_data_set))
    print(len(test_data_set))


    train_loader = DataLoader(train_data_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data_set, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data_set, batch_size=BATCH_SIZE, shuffle=False)
    print(len(train_loader), len(val_loader), len(test_loader))
    
    
    loss_fun = nn.L1Loss()
    # opt = optim.Adam(model.parameters(), lr=1e-3)
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    best_model = ""
    train_model(model, train_loader=train_loader, val_loader=val_loader, optimizer=opt, criterion=loss_fun, num_epochs=100,
                patience=10, save_path=weight_path)
    test_model(model=best_model.to(device), test_loader=test_loader)