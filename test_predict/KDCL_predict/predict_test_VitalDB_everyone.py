import torch
import sys
from InvertedResidual_small_Unet import invertedResidual_small_unet
from Model.InvertedResidual_large_Unet import invertedResidual_large_unet
from Model.U_Net_1D import UNet
sys.path.append('../')
from Model.Teacher import Teacher_Model
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import pandas as pd
import sys
import argparse
import h5py
from scipy.io import savemat
from tqdm import tqdm

from matlab_utils import matlab_cmd
import matlab.engine
from bp_analysis import get_SBP_DBP_Val


class MyDataset(Dataset):

    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):
        return torch.Tensor(np.array(self.x_data[index])), torch.Tensor(
            np.array(self.y_data[index]))

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder created: {folder_path}")
    else:
        print(f"Folder already exists: {folder_path}")


if __name__ == "__main__":
    
    # eng = matlab.engine.start_matlab()
    
    parser = argparse.ArgumentParser()



    # 添加命令行参数
    parser.add_argument('-fold_num', type=str)
    parser.add_argument('-res_column', type=str)
    parser.add_argument('-batch_size', type=int, default=256)
    parser.add_argument('-cuda_num', type=str, default='0')
    parser.add_argument('-save_path', type=str, default='result_data_UtransBPNet_VitalDB_everyone_external_data_Fold')
    parser.add_argument('-teacher_path', type=str)
    parser.add_argument('-excel_path', type=str, required=True, help='Path to the Excel file')
    # parser.add_argument('-excel_path', type=str)

    # 解析命令行参数
    args = parser.parse_args()

    # 获取参数值
    fold_number = args.fold_num
    RES_COLUMN = args.res_column
    BATCH_SIZE = args.batch_size
    cuda_number = args.cuda_num
    result_dir = args.save_path
    Teacher_Path = args.teacher_path
    EXCEL_PATH = args.excel_path
    # EXCEL_PATH = args.excel_path

    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_number
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    print(fold_number)
    # print(type(fold_number))
    file_path = '/home/yonghu/data_yonghu/code_python/VitalDB_dataset/VitalDB_Ten_0927/Group'+fold_number+'_in_out_split_everyone.h5'
    data_save_path = result_dir + fold_number + "/"
    create_folder_if_not_exists(data_save_path)
    column_name = 'Fold' + fold_number
    model_name = RES_COLUMN
    
    
    model = invertedResidual_small_unet().to(device)
    # model = invertedResidual_large_unet().to(device)
    # model = UNet().to(device)
    # model = Teacher_Model().to(device)
    checkpoint = torch.load(Teacher_Path)

    # 仅从checkpoint中提取模型状态字典
    model_state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    # model_list[1].load_state_dict(torch.load(Teacher_Path))
    model.load_state_dict(model_state_dict, strict=False)
    # model.load_state_dict(torch.load(Teacher_Path))
    peeson_result_loss = {}
    pred_li = []
    true_li = []
    
    UTN_ABP_MAE = {}
    UTN_SBP_MAE = {}
    UTN_DBP_MAE = {}
    UTN_MBP_MAE = {}
    UTN_ABP_MAE_pearson_crr = {}
    UTN_SBP_MAE_pearson_crr = {}
    UTN_DBP_MAE_pearson_crr = {}
    UTN_MBP_MAE_pearson_crr = {}
    
    ABP_true_list_tol = []
    SBP_true_list_tol = []
    DBP_true_list_tol = []
    MBP_true_list_tol = []
    
    ABP_pred_list_tol = []
    SBP_pred_list_tol = []
    DBP_pred_list_tol = []
    MBP_pred_list_tol = []
    
    ABP_pearson_list_tol = []
    SBP_pearson_list_tol = []
    DBP_pearson_list_tol = []
    MBP_pearson_list_tol = []
    

    
    with h5py.File(file_path, 'r') as file:
        # print("文件中的组：", list(file.keys()))
        print("人数：", len(file.keys()))

        df_all_persons = []  # 用于保存每个受试者的 SBP 数据

        for id2, person_name in enumerate(file.keys()):
            # if id2==3: break
            in_dataset = file[person_name]
            # print(in_dataset.keys())
            for person_info in in_dataset.keys():
                in_data = in_dataset[person_info]
                if person_info == 'In_Signals':
                    input_data = in_data[()][:, :, ::2]
                elif person_info == 'Out_Signals':
                    output_data = in_data[()][:, :, ::2]
                elif person_info == 'SegmentId':
                    SegmentId_data = in_data[()]

            x_test_data = torch.tensor(input_data)
            y_test_data = torch.tensor(output_data)

            test_data_set = MyDataset(x_test_data, y_test_data)
            test_data_loader = DataLoader(
                test_data_set, batch_size=BATCH_SIZE, shuffle=False)
            y_true = []
            out = []
            x_ecg_li = []
            x_ppg_li = []
            x_vppg_li = []
            x_appg_li = []

            with torch.no_grad():
                model.eval()
                for z, (x, y) in enumerate((test_data_loader)):
                    # if z==10:break
                    x_ecg = x[:, 0, :]
                    x_ppg = x[:, 1, :]
                    x_vppg = x[:, 2, :]
                    x_appg = x[:, 3, :]
                    
                    # x_ecg = x[:, 0, :].flatten()
                    # x_ppg = x[:, 1, :].flatten()
                    # x_vppg = x[:, 2, :].flatten()
                    # x_appg = x[:, 3, :].flatten()
                    
                    # print(x_ecg.shape)
                    # print(x_ppg.shape)
                    # print(x_vppg.shape)
                    # print(x_appg.shape)
                    
                    input_data = x.to(device)
                    y = y.to(device)
                    temp = model(input_data)

                    out_1 = temp.detach().cpu().numpy()[:,0,:]
                    y_true_1 = y.detach().cpu().numpy()[:,0,:]
                    out.append(out_1)
                    y_true.append(y_true_1)
                    x_ecg_li.append(x_ecg)
                    x_ppg_li.append(x_ppg)
                    x_vppg_li.append(x_vppg)
                    x_appg_li.append(x_appg)

                out = np.concatenate(out)
                y_true = np.concatenate(y_true)
                x_ecg_arr = np.concatenate(x_ecg_li)
                x_ppg_arr = np.concatenate(x_ppg_li)
                x_vppg_arr = np.concatenate(x_vppg_li)
                x_appg_arr = np.concatenate(x_appg_li)
                
                # print(out.shape)
                # print(y_true.shape)
                # print(x_ecg_arr.shape)
                # print(x_ppg_arr.shape)
                # print(x_vppg_arr.shape)
                # print(x_appg_arr.shape)
                ABP_True_data = y_true
                ABP_Pred_data = out
                SBP_true_list = []
                DBP_true_list = []
                MBP_true_list = []
                for sub_arr in ABP_True_data:
                    # eng.workspace['ABP_Wave'] = matlab.double(sub_arr.tolist())
                    # SBP_t, DBP_t = matlab_cmd(eng.eval, "get_SBP_DBP_Val(ABP_Wave);", nargout=2)
                    SBP_t, DBP_t = get_SBP_DBP_Val(sub_arr)
                    MBP_t = (1 / 3) * SBP_t + (2 / 3) * DBP_t
                    SBP_true_list.append(SBP_t)
                    DBP_true_list.append(DBP_t)
                    MBP_true_list.append(MBP_t)
                
                SBP_pred_list = []
                DBP_pred_list = []
                MBP_pred_list = []
                for sub_arr in ABP_Pred_data:
                    # eng.workspace['ABP_Wave'] = matlab.double(sub_arr.tolist())
                    # SBP_t, DBP_t = matlab_cmd(eng.eval, "get_SBP_DBP_Val(ABP_Wave);", nargout=2)
                    SBP_t, DBP_t = get_SBP_DBP_Val(sub_arr)
                    MBP_t = (1 / 3) * SBP_t + (2 / 3) * DBP_t
                    SBP_pred_list.append(SBP_t)
                    DBP_pred_list.append(DBP_t)
                    MBP_pred_list.append(MBP_t)
                SBP_true_list = np.array(SBP_true_list)
                DBP_true_list = np.array(DBP_true_list)
                MBP_true_list = np.array(MBP_true_list)
                SBP_pred_list = np.array(SBP_pred_list)
                DBP_pred_list = np.array(DBP_pred_list)
                MBP_pred_list = np.array(MBP_pred_list)
                
                err_abp_mae = np.mean(np.abs(ABP_True_data.flatten() - ABP_Pred_data.flatten()))
                UTN_ABP_MAE[person_name] = err_abp_mae
                
                corr_matrix = np.corrcoef(ABP_True_data.flatten(), ABP_Pred_data.flatten())
                pearson_corr = corr_matrix[0, 1]
                UTN_ABP_MAE_pearson_crr[person_name] = pearson_corr
                ABP_pearson_list_tol.append(pearson_corr)
                
                
                sbp_t_mae = np.mean(np.abs(SBP_true_list - SBP_pred_list))
                dbp_t_mae = np.mean(np.abs(DBP_true_list - DBP_pred_list))
                mbp_t_mae = np.mean(np.abs(MBP_true_list - MBP_pred_list))
                #! UTN_SBP_MAE[person_name] = sbp_t_mae
                UTN_SBP_MAE[person_name] = sbp_t_mae
                UTN_DBP_MAE[person_name] = dbp_t_mae
                UTN_MBP_MAE[person_name] = mbp_t_mae
                
                corr_matrix = np.corrcoef(SBP_true_list, SBP_pred_list)
                pearson_corr = corr_matrix[0, 1]
                UTN_SBP_MAE_pearson_crr[person_name] = pearson_corr
                SBP_pearson_list_tol.append(pearson_corr)
                
                corr_matrix = np.corrcoef(DBP_true_list, DBP_pred_list)
                pearson_corr = corr_matrix[0, 1]
                UTN_DBP_MAE_pearson_crr[person_name] = pearson_corr
                DBP_pearson_list_tol.append(pearson_corr)
                
                corr_matrix = np.corrcoef(MBP_true_list, MBP_pred_list)
                pearson_corr = corr_matrix[0, 1]
                UTN_MBP_MAE_pearson_crr[person_name] = pearson_corr
                MBP_pearson_list_tol.append(pearson_corr)
                
                ABP_true_list_tol.append(ABP_True_data)
                ABP_pred_list_tol.append(ABP_Pred_data)
                SBP_true_list_tol.append(SBP_true_list)
                DBP_true_list_tol.append(DBP_true_list)
                MBP_true_list_tol.append(MBP_true_list)
                SBP_pred_list_tol.append(SBP_pred_list)
                DBP_pred_list_tol.append(DBP_pred_list)
                MBP_pred_list_tol.append(MBP_pred_list)
                
                
                # mae = np.mean(np.abs(out - y_true))
                savemat(data_save_path+person_name+"_MAE_"+str(err_abp_mae)+".mat",
                        {'SegmentId': SegmentId_data, 'ECG': x_ecg_arr, 'PPG': x_ppg_arr,  'VPPG': x_vppg_arr, 'APPG': x_appg_arr, 'ABP_True': y_true, 'ABP_Pred': out,
                         'SBP_true':SBP_true_list, 'DBP_true':DBP_true_list, 'MBP_true':MBP_true_list, 'SBP_pred':SBP_pred_list, 'DBP_pred':DBP_pred_list, 'MBP_pred':MBP_pred_list
                         })
                
                from datetime import datetime
                now = datetime.now()
                formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
                print(id2, "MAE = ", err_abp_mae, "     ", person_name, "       ", formatted_now)

            # tol_loss = test_model(model, test_data_loader)
            peeson_result_loss[person_name] = err_abp_mae


     # -------------------------------------------------------------------------------
    ABP_pred_list_tol = np.concatenate(ABP_pred_list_tol).flatten()
    ABP_true_list_tol = np.concatenate(ABP_true_list_tol).flatten()
    SBP_pred_list_tol = np.concatenate(SBP_pred_list_tol).flatten()
    SBP_true_list_tol = np.concatenate(SBP_true_list_tol).flatten()
    DBP_pred_list_tol = np.concatenate(DBP_pred_list_tol).flatten()
    DBP_true_list_tol = np.concatenate(DBP_true_list_tol).flatten()
    MBP_pred_list_tol = np.concatenate(MBP_pred_list_tol).flatten()
    MBP_true_list_tol = np.concatenate(MBP_true_list_tol).flatten()
    
    ABP_wave_tol_MAE = np.mean(np.abs(ABP_pred_list_tol - ABP_true_list_tol))
    ABP_wave_tol_Mean = np.mean(ABP_pred_list_tol - ABP_true_list_tol)
    ABP_wave_tol_std = np.std(ABP_pred_list_tol - ABP_true_list_tol)
    SBP_tol_MAE = np.mean(np.abs(SBP_pred_list_tol - SBP_true_list_tol))
    SBP_tol_Mean = np.mean(SBP_pred_list_tol - SBP_true_list_tol)
    SBP_tol_std = np.std(SBP_pred_list_tol - SBP_true_list_tol)
    DBP_tol_MAE = np.mean(np.abs(DBP_pred_list_tol - DBP_true_list_tol))
    DBP_tol_Mean = np.mean(DBP_pred_list_tol - DBP_true_list_tol)
    DBP_tol_std = np.std(DBP_pred_list_tol - DBP_true_list_tol)
    MBP_tol_MAE = np.mean(np.abs(MBP_pred_list_tol - MBP_true_list_tol))
    MBP_tol_Mean = np.mean(MBP_pred_list_tol - MBP_true_list_tol)
    MBP_tol_std = np.std(MBP_pred_list_tol - MBP_true_list_tol)
    
    data_dict = {}
    data_dict['ABP_wave_tol_MAE'] = ABP_wave_tol_MAE
    data_dict['ABP_wave_tol_Mean'] = ABP_wave_tol_Mean
    data_dict['ABP_wave_tol_std'] = ABP_wave_tol_std
    data_dict['SBP_tol_MAE'] = SBP_tol_MAE
    data_dict['SBP_tol_Mean'] = SBP_tol_Mean
    data_dict['SBP_tol_std'] = SBP_tol_std
    data_dict['DBP_tol_MAE'] = DBP_tol_MAE
    data_dict['DBP_tol_Mean'] = DBP_tol_Mean
    data_dict['DBP_tol_std'] = DBP_tol_std
    data_dict['MBP_tol_MAE'] = MBP_tol_MAE
    data_dict['MBP_tol_Mean'] = MBP_tol_Mean
    data_dict['MBP_tol_std'] = MBP_tol_std
    
    data_dict['ABP_wave_pearson_crr_MAE'] = np.mean(np.abs(ABP_pearson_list_tol))
    data_dict['ABP_wave_pearson_crr_Mean'] = np.mean(ABP_pearson_list_tol)
    data_dict['ABP_wave_pearson_crr_std'] = np.std(ABP_pearson_list_tol)
    data_dict['SBP_pearson_crr_MAE'] = np.mean(np.abs(SBP_pearson_list_tol))
    data_dict['SBP_pearson_crr_Mean'] = np.mean(SBP_pearson_list_tol)
    data_dict['SBP_pearson_crr_std'] = np.std(SBP_pearson_list_tol)
    data_dict['DBP_pearson_crr_MAE'] = np.mean(np.abs(DBP_pearson_list_tol))
    data_dict['DBP_pearson_crr_Mean'] = np.mean(DBP_pearson_list_tol)
    data_dict['DBP_pearson_crr_std'] = np.std(DBP_pearson_list_tol)
    data_dict['MBP_pearson_crr_MAE'] = np.mean(np.abs(MBP_pearson_list_tol))
    data_dict['MBP_pearson_crr_Mean'] = np.mean(MBP_pearson_list_tol)
    data_dict['MBP_pearson_crr_std'] = np.std(MBP_pearson_list_tol)




    excel_file_path = EXCEL_PATH
    # # 检查文件是否存在
    # if not os.path.exists(excel_file_path):
    #     print(f"Excel file '{excel_file_path}' does not exist. Creating a new one...")
    #     df = pd.DataFrame()
    #     df.to_excel(excel_file_path, index=False)
    

    sheet_name_tol = 'Sheet1'
    all_tables = pd.read_excel(EXCEL_PATH, sheet_name=None)
    # # 遍历所有表格，填充 "UTransBPNet" 表
    # for table_name, df in all_tables.items():
    #     if table_name == sheet_name_tol:
    #         # 填充多个 RES_COLUMN 列
    #         for metric, mapping in [
    #             ('ABP_MAE', UTN_ABP_MAE),
    #             ('SBP_MAE', UTN_SBP_MAE),
    #             ('DBP_MAE', UTN_DBP_MAE),
    #             ('MBP_MAE', UTN_MBP_MAE),
    #             ('ABP_Pearson_Crr', UTN_ABP_MAE_pearson_crr),
    #             ('SBP_Pearson_Crr', UTN_SBP_MAE_pearson_crr),
    #             ('DBP_Pearson_Crr', UTN_DBP_MAE_pearson_crr),
    #             ('MBP_Pearson_Crr', UTN_MBP_MAE_pearson_crr)
    #         ]:
    #             RES_COLUMN = model_name + f'_{metric}'
    #             new_data = df['SubjectID'].map(mapping)
    #             df[RES_COLUMN] = new_data  # 直接赋值，不需要 combine_first

    #     elif table_name == model_name:
    #         # 找到 "Fold01" 行并填充对应信息
    #         fold_row = df[df['Folders'] == column_name]
    #         if not fold_row.empty:
    #             df.loc[fold_row.index, data_dict.keys()] = data_dict.values()

    # # 将更新后的所有表格写回 Excel 文件
    # with pd.ExcelWriter(excel_file_path, engine='openpyxl') as writer:
    #     for table_name, df in all_tables.items():
    #         df.to_excel(writer, sheet_name=table_name, index=False)
    
    # 遍历所有表格，仅填充 "UTransBPNet" 表
    for table_name, df in all_tables.items():
        if table_name == sheet_name_tol:
            RES_COLUMN = model_name + '_ABP_MAE'
            new_data = df['SubjectID'].map(UTN_ABP_MAE)
            df[RES_COLUMN] = new_data.combine_first(df[RES_COLUMN])

            RES_COLUMN =  model_name + '_SBP_MAE'
            new_data = df['SubjectID'].map(UTN_SBP_MAE)
            df[RES_COLUMN] = new_data.combine_first(df[RES_COLUMN])

            RES_COLUMN =  model_name + '_DBP_MAE'
            new_data = df['SubjectID'].map(UTN_DBP_MAE)
            df[RES_COLUMN] = new_data.combine_first(df[RES_COLUMN])


            RES_COLUMN =  model_name + '_MBP_MAE'
            new_data = df['SubjectID'].map(UTN_MBP_MAE)
            df[RES_COLUMN] = new_data.combine_first(df[RES_COLUMN])


            RES_COLUMN =  model_name + '_ABP_Pearson_Crr'
            new_data = df['SubjectID'].map(UTN_ABP_MAE_pearson_crr)
            df[RES_COLUMN] = new_data.combine_first(df[RES_COLUMN])


            RES_COLUMN =  model_name + '_SBP_Pearson_Crr'
            new_data = df['SubjectID'].map(UTN_SBP_MAE_pearson_crr)
            df[RES_COLUMN] = new_data.combine_first(df[RES_COLUMN])


            RES_COLUMN =  model_name + '_DBP_Pearson_Crr'
            new_data = df['SubjectID'].map(UTN_DBP_MAE_pearson_crr)
            df[RES_COLUMN] = new_data.combine_first(df[RES_COLUMN])


            RES_COLUMN =  model_name + '_MBP_Pearson_Crr'
            new_data = df['SubjectID'].map(UTN_MBP_MAE_pearson_crr)
            df[RES_COLUMN] = new_data.combine_first(df[RES_COLUMN])

        elif table_name == model_name:

                fold_row = df[df['Folders'] == column_name]
                df.loc[fold_row.index, data_dict.keys()] = data_dict.values()
        

    with pd.ExcelWriter(EXCEL_PATH, engine='openpyxl') as writer:
        for table_name, df in all_tables.items():
            df.to_excel(writer, sheet_name=table_name, index=False)

    # eng.quit()
