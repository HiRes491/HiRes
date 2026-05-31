# HiRes

This repo contains the code for our paper **HiRes: A Hierarchical Cascaded Method for Resistor Value Identification**
<p align="center">
    <img width="3240" height="2209" alt="overview (1)" src="https://github.com/user-attachments/assets/8633de36-97ef-44ae-963d-0ed5a5564eba" />
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
