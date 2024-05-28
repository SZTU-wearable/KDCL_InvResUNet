import torch
from torch import nn
# from Teacher import Teacher_Model
from Teacher import Teacher_Model
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch import optim
import os
# from Utils_tool import EarlyStopping
# from Utils_tool_1 import load_data_drink
# from Utils_tool_1 import load_data_exercise
# from Utils_tool_1 import load_data_mimic_ii
import sys
# from transformers import AdamW
import torch.optim as optim
from datetime import datetime
import numpy as np
import h5py

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class MyDataset(Dataset):

    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):
        return torch.Tensor(np.array(self.x_data[index])), torch.Tensor(
            np.array(self.y_data[index]))

import plotly.graph_objects as go


def visualization(y1):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(y=y1,
                   mode="lines",
                   line=dict(color="blue"),
                   name="predict value"))
    fig.update_layout()
    fig.show()


def load_data_pulseDB(idx):
    x_tol_arr = []
    y_tol_arr = []
    for i in range(1, 11):
        # file_name = '/home/yonghu/data_yonghu/code_python/PulseDB_Ten_0728/Group%02d_in_out_split.h5' % i
        file_name = '/home/yonghu/data_yonghu/code_python/VitalDB_dataset/VitalDB_Ten_0927/Group%02d_in_out_split.h5' % i
        print(file_name)
        with h5py.File(file_name, "r") as file:
            for dataset_name in file:
                if dataset_name=='In_Signals':
                    In_Signals = file[dataset_name][()][:,:,::2]
                    print(In_Signals.shape)
                    x_tol_arr.append(In_Signals)
                elif dataset_name=='Out_Signals':
                    Out_Signals = file[dataset_name][()][:,:,::2]
                    print(Out_Signals.shape)
                    y_tol_arr.append(Out_Signals)
            
    x_train_load_data = x_tol_arr[7:10] + x_tol_arr[0:5]
    x_val_load_data = x_tol_arr[6:7]
    x_test_load_data = x_tol_arr[5:6]
    x_train_load_data = np.vstack(x_train_load_data)
    x_val_load_data = np.vstack(x_val_load_data)
    x_test_load_data = np.vstack(x_test_load_data)
    print(x_train_load_data.shape)
    print(x_val_load_data.shape)
    print(x_test_load_data.shape)
    y_train_load_data = y_tol_arr[7:10] + y_tol_arr[0:5]
    y_val_load_data = y_tol_arr[6:7]
    y_test_load_data = y_tol_arr[5:6]
    y_train_load_data = np.vstack(y_train_load_data)
    y_val_load_data = np.vstack(y_val_load_data)
    y_test_load_data = np.vstack(y_test_load_data)
    print(y_train_load_data.shape)
    print(y_val_load_data.shape)
    print(y_test_load_data.shape)

    # y_test_load_data = y_test_load_data / 300

    # x_train_load_data = x_train_load_data[0:1000]
    # y_train_load_data = y_train_load_data[0:1000]
    # x_val_load_data = x_val_load_data[0:1000]
    # y_val_load_data = y_val_load_data[0:1000]
    # x_test_load_data = x_test_load_data[0:1000]
    # y_test_load_data = y_test_load_data[0:1000]

    x_train_data = torch.tensor(x_train_load_data)
    y_train_data = torch.tensor(y_train_load_data)
    x_val_data = torch.tensor(x_val_load_data)
    y_val_data = torch.tensor(y_val_load_data)
    x_test_data = torch.tensor(x_test_load_data)
    y_test_data = torch.tensor(y_test_load_data)

    train_data_set = MyDataset(x_train_data, y_train_data)
    val_data_set = MyDataset(x_val_data, y_val_data)
    test_data_set = MyDataset(x_test_data, y_test_data)

    return train_data_set, val_data_set, test_data_set

train_data_set, val_data_set, test_data_set = load_data_pulseDB("01")

train_loader = DataLoader(train_data_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data_set, batch_size=64, shuffle=False)
test_loader = DataLoader(test_data_set, batch_size=64, shuffle=False)
print(len(train_loader), len(val_loader), len(test_loader))

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

weight_path = "weight_model/UtransBPNet_VitalDB_Fold06_"
model = Teacher_Model(4, 128, 7, 3).to(device)

loss_fun = nn.L1Loss()
# opt = optim.Adam(model.parameters(), lr=1e-3)
opt = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.1)


train_model(model, train_loader=train_loader, val_loader=val_loader, optimizer=opt, criterion=loss_fun, num_epochs=100,
            patience=5, save_path=weight_path)

# test_model(model=best_model.to(device), test_loader=test_loader)