<div align="center">

<p align="left">
  <img src="logo.png" >
</p>

Generalized-YOLOv5 is a modified version of YOLOv5.
Two main contributions have been made to Generalized-YOLOv5. First, an extension to train on non-natural images and second a cross-validation extension.

### Non-Natural image extension
The non-natural image extension enables YOLOv5 to handle images with arbitrary intensity scales, so it can be trained on 2D non-natural images.
Furthermore, this extension also includes preprocessing scripts to convert non-natural image datasets to a natural intensity scale.
Training on natural images is faster as OpenCV, which is highly optimized on natural images, can be used during training for preprocessing and augmentation.
However, this can lead in some scenarios to a decreased performance as the intensity scale is often reduced to a fraction of its original scale.
The slow-down on non-natural images can be somewhat mitigated by increasing worker threads (until they block each other), but one should expect an increased training time of 1.5x.

### N-fold cross-validation
The cross-validation extension is a dataset preprocessing script that processes a dataset into N folds with corresponding dataset configuration files that are understood by YOLOv5.
On each fold a YOLOv5 model can be trained and YOLOv5's built-in ensembling method can be used to run inference with the models of all folds.
The built-in ensembling method itself has no cross-validation functionality and was only designed for ad-hoc ensembling of YOLOv5 models of different scales, thus the preprocessing script is required for cross-validation.

## <div align="center">Documentation</div>

See the documentation and usage of the Generalized-YOLOv5 in the [Generalized-YOLOv5 README](evaluations/Node21/README.md).
The official YOLOv5 documentation can be found in the [YOLOv5 README](README_YOLOV5.md).

## <div align="center">Contact</div>

For Generalized-YOLOv5 bugs and feature requests please visit [GitHub Issues](https://github.com/ultralytics/yolov5/issues). For business inquiries or
professional support requests please visit [https://ultralytics.com/contact](https://ultralytics.com/contact).

<br>

<div align="center">
    <a href="https://github.com/ultralytics">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-github.png" width="3%"/>
    </a>
    <img width="3%" />
    <a href="https://www.linkedin.com/company/ultralytics">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-linkedin.png" width="3%"/>
    </a>
    <img width="3%" />
    <a href="https://twitter.com/ultralytics">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-twitter.png" width="3%"/>
    </a>
    <img width="3%" />
    <a href="https://youtube.com/ultralytics">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-youtube.png" width="3%"/>
    </a>
    <img width="3%" />
    <a href="https://www.facebook.com/ultralytics">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-facebook.png" width="3%"/>
    </a>
    <img width="3%" />
    <a href="https://www.instagram.com/ultralytics/">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-instagram.png" width="3%"/>
    </a>
</div>
