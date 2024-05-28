model_name="UNet"

batch_size="256"

cuda_num="1"

python train_Ten_Fold.py -model_name "$model_name" -fold_num 01 -batch_size "$batch_size" -cuda_num "$cuda_num"

python train_Ten_Fold.py -model_name "$model_name" -fold_num 02 -batch_size "$batch_size" -cuda_num "$cuda_num"

python train_Ten_Fold.py -model_name "$model_name" -fold_num 03 -batch_size "$batch_size" -cuda_num "$cuda_num"

python train_Ten_Fold.py -model_name "$model_name" -fold_num 04 -batch_size "$batch_size" -cuda_num "$cuda_num"

python train_Ten_Fold.py -model_name "$model_name" -fold_num 05 -batch_size "$batch_size" -cuda_num "$cuda_num"

python train_Ten_Fold.py -model_name "$model_name" -fold_num 06 -batch_size "$batch_size" -cuda_num "$cuda_num"

python train_Ten_Fold.py -model_name "$model_name" -fold_num 07 -batch_size "$batch_size" -cuda_num "$cuda_num"

python train_Ten_Fold.py -model_name "$model_name" -fold_num 08 -batch_size "$batch_size" -cuda_num "$cuda_num"

python train_Ten_Fold.py -model_name "$model_name" -fold_num 09 -batch_size "$batch_size" -cuda_num "$cuda_num"

python train_Ten_Fold.py -model_name "$model_name" -fold_num 10 -batch_size "$batch_size" -cuda_num "$cuda_num"
