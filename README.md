# KDCL_InvResUNet

- train_UNet_Ten_Folds: independent training of UNet.
- kdcl_train_allmodel_Ten_Folds_4_mean: Collaborative learning with mean value as the second-level representation.
- kdcl_train_allmodel_Ten_Folds_4_min: Collaborative learning with the model with minimal error as the second-level representation.
- train_UTransBPNet_VitalDB_Ten_Folds:Independent training of UTransBPNet.
- train_InvertedResidual_small_Unet_Ten_Folds: Independent training of sInvResUNet.
- train_InvertedResidual_large_Unet_Ten_Folds: Independent training of lInvResUNet.
- Model:Descriptions of all model structures used in the training above.
- Embedded_deployment_code:Code deployed to the Raspberry Pi 4 Model B.
- Ablation:experiments without Unet or without UTransBPnet in Collaborative learning, and without the SE block in sInvResUNet.
- test_predict:This directory contains test prediction scripts.
