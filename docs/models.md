# Model provenance and licenses

| Component | Upstream | Code license / notes |
|---|---|---|
| PyTorch | https://github.com/pytorch/pytorch | BSD-style; see upstream LICENSE |
| TorchVision and three image architectures | https://github.com/pytorch/vision | BSD-3-Clause code |
| scikit-learn classical models | https://github.com/scikit-learn/scikit-learn | BSD-3-Clause |
| XGBoost | https://github.com/dmlc/xgboost | Apache-2.0 |
| timm | https://github.com/huggingface/pytorch-image-models | Apache-2.0 code; individual weights have separate provenance |
| statsmodels | https://github.com/statsmodels/statsmodels | BSD-3-Clause |
| SHAP | https://github.com/shap/shap | MIT |
| Grad-CAM | https://github.com/jacobgil/pytorch-grad-cam | MIT |

Official checkpoint references:

- [EfficientNet-B0](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html), IMAGENET1K_V1.
- [ConvNeXt-Tiny](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_tiny.html), IMAGENET1K_V1.
- [MobileNetV3-Large](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_large.html), IMAGENET1K_V2.
- [PLS regression](https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html).
- [XGBoost API](https://xgboost.readthedocs.io/en/stable/python/python_api.html).

The architecture implementations are open source. This project's license does
not relicense third-party code, weights, or ImageNet data. TorchVision's model
documentation notes that pretrained weights may have separate terms inherited
from training data. Check upstream terms for the intended redistribution/use.
The downloaded weights remain local; only their source manifest is versioned.

The spectral and fusion models require study-specific training; there are no
downloadable pretrained weights for this proposed paired dataset. No training
performance or scientific validity is implied by installation.
