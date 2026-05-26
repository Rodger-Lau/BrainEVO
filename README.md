# BrainEVO: A brain inspired evolving deep learning architecture for cross-domain adaptation
## Python version
  python 3.11
## Data sources
### NYC dataset
Taxi Trip Records https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page

Demographic Data: https://data.cityofnewyork.us/City-Government/Demograp-hic-Statistics-By-Zip-Code/

Road Network: https://data.cityofnewyork.us/City-Government/NYC-Street-Centerline-CSCL-/

### SIP dataset
Due to the privacy protocols between our department and SIP traffic administration offices, the statistics of traffic flows and speed values cannot be open source.

### Chicago dataset
Taxi Trip Records https://data.cityofchicago.org/Transportation/Taxi-Trips-2013-2023-/wrvz-psew/about_data

Traffic Crashes - People https://data.cityofchicago.org/Transportation/Traffic-Crashes-People/u6pd-qa9d/about_data

Traffic Crashes - Crashes https://data.cityofchicago.org/Transportation/Traffic-Crashes-Crashes/85ca-t3if/about_data

Traffic Crashes - Vehicles https://data.cityofchicago.org/Transportation/Traffic-Crashes-Vehicles/68nd-jvt3/about_data

### SD dataset

https://github.com/liuxu77/LargeST/tree/main/data/sd

### MUTAG dataset

https://grlplus.github.io/papers/79.pdf

### HIV dataset

https://hf-mirror.com/datasets/OGB/ogbg-molhiv

### BBBP dataset

https://moleculenet.org/datasets-1

## Data description
For feature x,   

    batch_size = x.shape[0]
    
    tod (time of day) = x[..., 1]
    
    dow (day of week) = x[..., 2]
    
    coor (coordinates)= x[..., 3:5]
    
    timestamp = x[..., 5:11], whose type is [year, month, date, hour, minute, second]

## Training example
First,

  put the processed data (divided into train.npz, val.npz and test.npz, where npz file is in shape(num_samples, length, num_nodes, dim)) under ./data/data/{dataset name}, e.g., ./data/data/SIP

Then run the following command(take SIP as an example),

    python main.py --dataset SIP
