# HiRes

This repo contains the code for our paper **HiRes: A Hierarchical Cascaded Method for Resistor Value Identification**
<p align="center">
<img width="654" height="375" alt="Screenshot 2026-05-31 at 6 03 21 PM" src="https://github.com/user-attachments/assets/5d67153a-6064-460b-b46d-6d360e59be0e" />
</p>  

## Installation 

- Clone the repository:
```bash
git clone https://github.com/HiRes491/HiRes.git
cd HiRes
```

- Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

- Install dependencies:
```bash
pip install -r requirements.txt
```

## Getting Started
Download the pretrained model weights and place them as follows:
```bash
weights/
├── detection/
│   └── best.pt
└── segmentation/
    └── efficientnet-b2_best.pt
```

### Quickstart (Notebook)
Open ```quickstart.ipynb```. The notebook walks through the full pipeline interactively:
- Load weights — loads the YOLOv8n detector and UNet++ segmentation model
- Run on a single image — set IMAGE_PATH to your image and run the pipeline cell
- Run on a folder — set FOLDER to a directory of images to batch-process and print results

### Full Pipeline
```pipeline.py``` runs the complete end-to-end pipeline from the command line and saves a 3-panel composite image per detected resistor alongside a results.txt summary.

- Single image:
```bash
python pipeline.py path/to/image.jpg
```
- Directory of images:
```bash
python pipeline.py path/to/images/
```
- Custom weights or output directory:
```bash
python pipeline.py path/to/image.jpg \
    --det_weights weights/detection/best.pt \
    --seg_weights weights/segmentation/efficientnet-b2_best.pt \
    --output_dir results/pipeline_output
```
