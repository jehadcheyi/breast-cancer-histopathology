---
title: Breast AI PRO - Histopathology Classifier V2
emoji: 🧠
colorFrom: pink
colorTo: red
sdk: gradio
sdk_version: 5.42.0
app_file: app.py
pinned: true
---

# Breast AI PRO

**Live demo:** [www.histopathology.ai](https://huggingface.co/spaces/jehadcheyi/breast-cancer-histopathology3)

A two-stage deep learning system for classifying breast histopathology images, with Grad-CAM explanations and an interface in English, Arabic and Kurdish.

I built this as part of my research on computational pathology. The goal is simple: given a histopathology image, tell the user whether the tissue looks benign or malignant, name the most likely subtype, and show which regions of the image drove that decision.

> **Research use only.** This tool is not a medical device and must not be used for clinical diagnosis or treatment decisions. Any output should be reviewed by a qualified pathologist.

## How it works

The system runs three models in sequence, each trained for one job.

1. **Binary classification.** A CNN decides whether the image is benign or malignant.
2. **Subtype classification.** Depending on the first result, the image is passed to one of two specialised models:
   - Benign: adenosis, fibroadenoma, phyllodes tumor, tubular adenoma
   - Malignant: ductal carcinoma, lobular carcinoma, mucinous carcinoma, papillary carcinoma
3. **Grad-CAM.** Heatmaps are generated from the final convolutional layers and overlaid on the original image, at four zoom levels (100%, 75%, 50%, 25%), so the reader can check whether the model is attending to clinically meaningful structures.

Reported accuracy is above 98% on [dataset and split: fill in]. Please treat this as a research result on that benchmark, not a guarantee of performance on images from other labs, scanners or staining protocols.

| Category  | Code | Subtype             |
|-----------|------|---------------------|
| Benign    | A    | Adenosis            |
| Benign    | F    | Fibroadenoma        |
| Benign    | PT   | Phyllodes tumor     |
| Benign    | TA   | Tubular adenoma     |
| Malignant | DC   | Ductal carcinoma    |
| Malignant | LC   | Lobular carcinoma   |
| Malignant | MC   | Mucinous carcinoma  |
| Malignant | PC   | Papillary carcinoma |

## Features

- Upload your own image (`.jpg`, `.jpeg`, `.png`, `.tif`) or pick one from the built-in `example/` gallery
- Class probabilities for both the binary and subtype stages
- Grad-CAM overlays at multiple zoom levels
- Interface text in English, Arabic and Kurdish
- Model weights pulled from the Hugging Face Hub and cached locally on first run

## Running it locally

Requirements: Python 3.7+ and TensorFlow 2.4 or later.

```bash
pip install tensorflow gradio opencv-python matplotlib numpy pillow huggingface-hub
```

The model files (`binary.h5`, `benign.h5`, `malignant.h5`) are downloaded from the Hugging Face Hub, so you need an access token. The app reads it from an environment variable named `jehad`. Once it is set, start the app with `python app.py`. In Colab or Jupyter, call `demo.launch(share=True)` instead to get a public link.

## Project structure

```
.
├── app.py          # Gradio app and inference pipeline
├── example/        # Sample histopathology images
├── model_cache/    # Models cached from the Hugging Face Hub
└── README.md
```

## Planned work

- Lesion segmentation
- Docker / FastAPI deployment
- Performance monitoring
- Mobile access
- Integration with electronic health records

## Limitations

- Trained and evaluated on a single public benchmark; external validation is still needed.
- Grad-CAM shows where the model looked, not why. It is an aid for inspection, not proof of correct reasoning.
- Image quality, magnification and staining differences can reduce accuracy.

## License

MIT. Use and modify freely for academic and research purposes.

## Acknowledgments

Built with TensorFlow, Gradio and the Hugging Face Hub. Sample images are adapted from public datasets. Thanks to the researchers and clinicians whose annotations made this work possible.

## Contact

Jehad Cheyi, Duhok, Kurdistan Region of Iraq.
For collaboration or questions, please open an issue on this repository or get in touch directly.
