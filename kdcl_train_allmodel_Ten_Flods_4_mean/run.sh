
batch_size="64"

cuda_num="1"

save_path="weight_model_4_utrans_unet_InSmUNet"

python train_Ten_Fold.py -fold_num 01 -batch_size "$batch_size" -cuda_num "$cuda_num" -save_path "$save_path"

python train_Ten_Fold.py -fold_num 02 -batch_size "$batch_size" -cuda_num "$cuda_num" -save_path "$save_path"

python train_Ten_Fold.py -fold_num 03 -batch_size "$batch_size" -cuda_num "$cuda_num" -save_path "$save_path"

python train_Ten_Fold.py -fold_num 04 -batch_size "$batch_size" -cuda_num "$cuda_num" -save_path "$save_path"

python train_Ten_Fold.py -fold_num 05 -batch_size "$batch_size" -cuda_num "$cuda_num" -save_path "$save_path"

python train_Ten_Fold.py -fold_num 06 -batch_size "$batch_size" -cuda_num "$cuda_num" -save_path "$save_path"

python train_Ten_Fold.py -fold_num 07 -batch_size "$batch_size" -cuda_num "$cuda_num" -save_path "$save_path"

python train_Ten_Fold.py -fold_num 08 -batch_size "$batch_size" -cuda_num "$cuda_num" -save_path "$save_path"

python train_Ten_Fold.py -fold_num 09 -batch_size "$batch_size" -cuda_num "$cuda_num" -save_path "$save_path"

python train_Ten_Fold.py -fold_num 10 -batch_size "$batch_size" -cuda_num "$cuda_num" -save_path "$save_path"

