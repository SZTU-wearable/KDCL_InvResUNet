column_name="InvertedResidual_UNet_small"

save_path="result_data_InvertedResidual_UNet_small_VitalDB_everyone"


EXCEL_PATH_NAME="KDCL_mean_InvertedResidual_UNet_small_VitalDB_result.xlsx"


batch_size="32"

cuda_num="1"

python predict_test_VitalDB_everyone.py \
-fold_num "01" \
-teacher_path "../../kdcl_train_allmodel_Ten_Flods_4-copy/weight_model_4_utrans_noUBPnet_InSmUNet/InvertedResidual_UNet_small/InvertedResidual_UNet_smallFold01_2025_01_24_18:02:26_testMAE_[9.604206763472087, 9.630285922449225, 9.808690382170807].pth" \
-excel_path "$EXCEL_PATH_NAME" \
-res_column "$column_name" \
-batch_size "$batch_size" \
-cuda_num "$cuda_num" \
-save_path "$save_path"

python predict_test_VitalDB_everyone.py \
-fold_num "02" \
-teacher_path "../../kdcl_train_allmodel_Ten_Flods_4-copy/weight_model_4_utrans_noUBPnet_InSmUNet/InvertedResidual_UNet_small/InvertedResidual_UNet_smallFold02_2025_01_25_09:22:30_testMAE_[10.231333993324537, 10.195609156728432, 10.2339700855075].pth" \
-excel_path "$EXCEL_PATH_NAME" \
-res_column "$column_name" \
-batch_size "$batch_size" \
-cuda_num "$cuda_num" \
-save_path "$save_path"

python predict_test_VitalDB_everyone.py \
-fold_num "03" \
-teacher_path "../../kdcl_train_allmodel_Ten_Flods_4-copy/weight_model_4_utrans_noUBPnet_InSmUNet/InvertedResidual_UNet_small/InvertedResidual_UNet_smallFold03_2025_01_25_19:31:09_testMAE_[9.929007514941967, 10.07891900839708, 10.100292329725503].pth" \
-excel_path "$EXCEL_PATH_NAME" \
-res_column "$column_name" \
-batch_size "$batch_size" \
-cuda_num "$cuda_num" \
-save_path "$save_path"

python predict_test_VitalDB_everyone.py \
-fold_num "04" \
-teacher_path "../../kdcl_train_allmodel_Ten_Flods_4-copy/weight_model_4_utrans_noUBPnet_InSmUNet/InvertedResidual_UNet_small/InvertedResidual_UNet_smallFold04_2025_01_26_03:04:25_testMAE_[10.349288186127811, 10.372045549377642, 10.42909872214673].pth" \
-excel_path "$EXCEL_PATH_NAME" \
-res_column "$column_name" \
-batch_size "$batch_size" \
-cuda_num "$cuda_num" \
-save_path "$save_path"

python predict_test_VitalDB_everyone.py \
-fold_num "05" \
-teacher_path "../../kdcl_train_allmodel_Ten_Flods_4-copy/weight_model_4_utrans_noUBPnet_InSmUNet/InvertedResidual_UNet_small/InvertedResidual_UNet_smallFold05_2025_01_26_15:17:13_testMAE_[9.853776353721111, 9.973655825934982, 10.84121457160865].pth" \
-excel_path "$EXCEL_PATH_NAME" \
-res_column "$column_name" \
-batch_size "$batch_size" \
-cuda_num "$cuda_num" \
-save_path "$save_path"

python predict_test_VitalDB_everyone.py \
-fold_num "06" \
-teacher_path "../../kdcl_train_allmodel_Ten_Flods_4-copy/weight_model_4_utrans_noUBPnet_InSmUNet/InvertedResidual_UNet_small/InvertedResidual_UNet_smallFold06_2025_01_26_22:44:54_testMAE_[10.27946532011774, 9.96158952010008, 10.101189654089684].pth" \
-excel_path "$EXCEL_PATH_NAME" \
-res_column "$column_name" \
-batch_size "$batch_size" \
-cuda_num "$cuda_num" \
-save_path "$save_path"

python predict_test_VitalDB_everyone.py \
-fold_num "07" \
-teacher_path "../../kdcl_train_allmodel_Ten_Flods_4-copy/weight_model_4_utrans_noUBPnet_InSmUNet/InvertedResidual_UNet_small/InvertedResidual_UNet_smallFold07_2025_01_27_06:33:27_testMAE_[10.176212104868279, 10.225579434779778, 10.244473562548047].pth" \
-excel_path "$EXCEL_PATH_NAME" \
-res_column "$column_name" \
-batch_size "$batch_size" \
-cuda_num "$cuda_num" \
-save_path "$save_path"

python predict_test_VitalDB_everyone.py \
-fold_num "08" \
-teacher_path "../../kdcl_train_allmodel_Ten_Flods_4-copy/weight_model_4_utrans_noUBPnet_InSmUNet/InvertedResidual_UNet_small/InvertedResidual_UNet_smallFold08_2025_01_27_13:45:41_testMAE_[9.577842365954586, 9.67068831276418, 9.604724795131167].pth" \
-excel_path "$EXCEL_PATH_NAME" \
-res_column "$column_name" \
-batch_size "$batch_size" \
-cuda_num "$cuda_num" \
-save_path "$save_path"

python predict_test_VitalDB_everyone.py \
-fold_num "09" \
-teacher_path "../../kdcl_train_allmodel_Ten_Flods_4-copy/weight_model_4_utrans_noUBPnet_InSmUNet/InvertedResidual_UNet_small/InvertedResidual_UNet_smallFold09_2025_01_28_02:45:57_testMAE_[10.07730904215487, 10.058683745210777, 10.362569794555311].pth" \
-excel_path "$EXCEL_PATH_NAME" \
-res_column "$column_name" \
-batch_size "$batch_size" \
-cuda_num "$cuda_num" \
-save_path "$save_path"

python predict_test_VitalDB_everyone.py \
-fold_num "10" \
-teacher_path "../../kdcl_train_allmodel_Ten_Flods_4-copy/weight_model_4_utrans_noUBPnet_InSmUNet/InvertedResidual_UNet_small/InvertedResidual_UNet_smallFold10_2025_01_28_11:56:07_testMAE_[9.801388828731243, 9.979242187406557, 9.863761894136848].pth" \
-excel_path "$EXCEL_PATH_NAME" \
-res_column "$column_name" \
-batch_size "$batch_size" \
-cuda_num "$cuda_num" \
-save_path "$save_path"
