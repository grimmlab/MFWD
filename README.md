# MFWD
Code repository for the manuscript "Manually annotated and curated dataset of diverse weed species in maize and sorghum for computer vision"


## Requirements
- ftplib
- pathlib
- pandas
- numpy
- argparse
- tqdm
- albumentations
- scikit-image
- scikit-learn
- timm
- matplotlib
- pytorch

## Usage Notes
### Download Data
Use the following code to download a part of the dataset.

a) All files annotated with segmentation masks 
```python
python3 download_by_ftp.py masks
```

b) All images of multiple specific species 
```python
python3 download_by_ftp.py species 'ACHMI, CHEAL'
```

Possible arguments: 
- save_path: path to save the files to
- files: comma separated list of EPPO codes or tray IDs (enclosed with "")
- img_type: Type of the images: either 'jpegs' for compressed jpeg images or 'pngs' for uncompressed images in PNG format

c) All images of multiple specific trays 
```python
python3 download_by_ftp.py trays '109801, 109802'
```

### Re-scale Data
As the resolution of our data might be too high for certain tasks, we provide the script `rescale_data.py` to change the resolution.

### Model Baseline
We provide a Dockerfile to train classification models using our data. The code is provided for reproducibility.
1) Run the script `prepare_data.py` to generate images of the plant cut-outs.
2) Run the script `optimize_hyperparameters_efficientnet.py` to reproduce the results on training the EfficientNet_b0 model architecture. 
3) Run the script `test.py` to generate the weighted f1-score on the test-set. 


## Citing

TBA