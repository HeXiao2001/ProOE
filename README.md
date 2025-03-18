# ProOE

This is the implementation of "A Probabilistic Optimal Estimation Method for Detecting Spatial Fuzzy Communities"


![Illustration of the core idea of the model](Core4model.png)


## Requirements
- python 3.12.5
- numpy
- os
- pandas
- geopandas
- matplotlib
- tqdm
- scipy
- mpl_toolkits
- kneed
- termcolor
- hdbscan
- time
- importlib
- argparse

## Running Examples

There are two ways to run the code:

1. (Recommended)You can run the `demo.ipynb` file, which includes the step-by-step instructions to reproduce findings reported in the manuscript. The processing and visualization of the New York City taxi dataset and three simulated datasets are organized in it. Visualizations include spatial fuzzy community division results, Trip Matrix, Confidence Index, and Certainty Index. 

2. Alternatively, you can run the `main.py` file. The New York City taxi dataset and three simulated datasets are also available, and you can easily switch between datasets.

The results are saved in the `./data/output` directory, organized by dataset name. You can open this directory to view the detailed data.

Please note that we tested the code on a PC with a 3.50 GHz CPU and 64 GB memory running the Windows 11 operating system. Different device conditions may affect the results.