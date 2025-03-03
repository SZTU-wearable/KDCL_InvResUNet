## Ablation Study Implementation

### KDCL_Model Variants

- To ablate KDCL by removing U-net and UtransBPnet:
  - Comment out corresponding model imports in `Model/_init_.py`

### sInvResUNet architecture Modifications

- To remove SE module from sInvResUNet (NoSE variant):
  - Set `use_se = 0` in `InvertedResidual_small_Unet.py`
