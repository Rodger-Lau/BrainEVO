# BrainEVO: A brain inspired evolving deep learning architecture for cross-domain adaptation
## Python version
  python 3.11
## Data sources
### NYC dataset
Taxi Trip Records https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page

Demographic Data: https://data.cityofnewyork.us/City-Government/Demograp-hic-Statistics-By-Zip-Code/

Road Network: https://data.cityofnewyork.us/City-Government/NYC-Street-Centerline-CSCL-/

The processed data can be found in https://drive.google.com/drive/folders/1OTMUKbQjfUOQNqXQ-MsUVy84sn5JCfda?usp=sharing

### SIP dataset
The processed data can be found in https://drive.google.com/drive/folders/1jSaVS3b7Wm7qyA1C2B619eX_yA6zRBLF?usp=sharing

### Chicago dataset
Taxi Trip Records https://data.cityofchicago.org/Transportation/Taxi-Trips-2013-2023-/wrvz-psew/about_data

Traffic Crashes - People https://data.cityofchicago.org/Transportation/Traffic-Crashes-People/u6pd-qa9d/about_data

Traffic Crashes - Crashes https://data.cityofchicago.org/Transportation/Traffic-Crashes-Crashes/85ca-t3if/about_data

Traffic Crashes - Vehicles https://data.cityofchicago.org/Transportation/Traffic-Crashes-Vehicles/68nd-jvt3/about_data

The processed data can be found in https://drive.google.com/drive/folders/17wv8R5H8o2NGBaSv59QZFxeFY2LlPDPU?usp=sharing

### SD dataset

https://github.com/liuxu77/LargeST/tree/main/data/sd

### MUTAG dataset

https://grlplus.github.io/papers/79.pdf

### HIV dataset

https://hf-mirror.com/datasets/OGB/ogbg-molhiv

### BBBP dataset

https://moleculenet.org/datasets-1


## Training example
First,

  put the data under ./data/data/{dataset name}, e.g., ./data/data/SIP

Then run the following command(take SIP and BBBP as an example),

    python main_ST.py --dataset SIP
    python main_molecule.py --dataset BBBP

The implementation of baselines mentioned in revisions can be found in baseline_*.py
