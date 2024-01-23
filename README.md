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
Genze, N., Vahl, W.K., Groth, J. et al. Manually annotated and curated Dataset of diverse Weed Species in Maize and Sorghum for Computer Vision. Sci Data 11, 109 (2024). https://doi.org/10.1038/s41597-024-02945-6

@article{Genze2024,
   abstract = {Sustainable weed management strategies are critical to feeding the world’s population while preserving ecosystems and biodiversity. Therefore, site-specific weed control strategies based on automation are needed to reduce the additional time and effort required for weeding. Machine vision-based methods appear to be a promising approach for weed detection, but require high quality data on the species in a specific agricultural area. Here we present a dataset, the Moving Fields Weed Dataset (MFWD), which captures the growth of 28 weed species commonly found in sorghum and maize fields in Germany. A total of 94,321 images were acquired in a fully automated, high-throughput phenotyping facility to track over 5,000 individual plants at high spatial and temporal resolution. A rich set of manually curated ground truth information is also provided, which can be used not only for plant species classification, object detection and instance segmentation tasks, but also for multiple object tracking.},
   author = {Nikita Genze and Wouter K Vahl and Jennifer Groth and Maximilian Wirth and Michael Grieb and Dominik G Grimm},
   doi = {10.1038/s41597-024-02945-6},
   issn = {2052-4463},
   issue = {1},
   journal = {Scientific Data},
   pages = {109},
   title = {Manually annotated and curated Dataset of diverse Weed Species in Maize and Sorghum for Computer Vision},
   volume = {11},
   url = {https://doi.org/10.1038/s41597-024-02945-6},
   year = {2024},
}
