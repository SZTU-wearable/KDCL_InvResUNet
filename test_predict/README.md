# Test Prediction

This directory contains test prediction scripts for different model architectures and training approaches.

## Directory Structure

- `KDCL_predict/`: Scripts for testing models trained with Knowledge Distillation Collaborative Learning
- `Single_model_predict/`: Scripts for testing individually trained models

## KDCL_predict

This folder contains scripts for evaluating models trained using the Knowledge Distillation Collaborative Learning approach:

- `predict_test_VitalDB_everyone.py`: Python script for testing KDCL-trained models
- `predict_test_VitalDB_everyone.sh`: Shell script for batch testing execution

The prediction process loads trained model weights from the training phase and performs inference on the test dataset. The collaborative learning approach combines knowledge from multiple model architectures during training.

## Single_model_predict

This folder contains Jupyter notebooks for testing individual model architectures:

- `predict_std_InvResUNet_small.ipynb`: Testing script for small Inverse Residual UNet
- `predict_std_InvResUNet_large.ipynb`: Testing script for large Inverse Residual UNet
- `predict_std_UTransBPNet.ipynb`: Testing script for UTransBPNet
- `predict_std_UNet.ipynb`: Testing script for standard UNet

Each notebook loads the corresponding model weights saved during the individual training phase and performs prediction on the test dataset.

## Usage

1. Ensure the trained model weights are available in the specified path
2. For KDCL prediction:
   - Use the shell script to run batch predictions
   - Or execute the Python script directly for specific test cases
3. For single model prediction:
   - Open the corresponding notebook for your model architecture
   - Follow the notebook instructions to load weights and perform prediction

The prediction results will be saved according to the configuration specified in each script.
