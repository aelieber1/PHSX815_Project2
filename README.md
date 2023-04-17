# PHSX815_Project2

## Simple Simulation of Soccer Goals Using a Poisson Distribution with Gamma Distribution Priors & Hypothesis Testing

### This repository contains several programs:

- `GoalDataGeneration.py` [Python]
- `GoalDataAnalysis.py` [Python]
- `GammaGraphComparison.py` [Pytho
n]
- `MySort.py` [Python]
- `Random.py` [Python]

### Requirements

The Python code in this repository requires the use of several packages which can be 
easily installed from the command line using `HomeBrew` or `pip install` commands. 

In order to compile the program `GoalData.py` and `Random.py`, these external 
packages are required:
- `numpy`
- `scipy.stats`
    - `from scipy.stats import poisson`
    - `from scipy.stats import gamma`
- `math`

In order to compile the programs `GoalDataAnalysis.py` and `MySort.py`, these external 
packages are required:
- `math`
- `numpy`
- `matplotlib.pyplot`
- `pandas`
- `scipy.stats` import `poisson`
- `scipy.stats` import `gamma`

In order to compile the programs `GoalDataAnalysis.py` and `MySort.py`, these external 
packages are required:
- `numpy`

### Usage

The python file `GoalData.py` which simulates the experiment can be run from the command
line by typing:

	<> python3 GoalData.py -rate [rate] -seed [seed] -Nmeas [number of games observed] -Nexp [number of sets of measurments or seasons observed] -output ["filename"]

This script will either print the result to the command line or save to a file if given a filename from the command line.

FIX FIX FIX FIX 
The other files in the repository `H0_data_lam_3_10000.txt`, `H1_data_lam_5_10000.txt`, `H1_data_lam_9_10000.txt` are examples of data textfiles that were the output of running `GoalData.py` under different rate parameters for 20 measurements per 10000 experiments. 


With the two datasets on hand from the previous script, the next python file to run is `GoalDataAnalysis.py`  which can be run from the commandline by typing:

	<> python3 GoalData.py -inputH0 ["filename"] -inputH1 ["filename"]

This script will conduct our analysis and hypothesis testing of these two datasets and output two plots, (1) a histogram of the data provided from both hypotheses, and (2) a log likelihood ratio plot which includes the critical lambda value and the final power of the test calculated. 

### Other Notes

- All of the Python programs can be called from the command line with the `-h` 
or `--help` flag, which will print the options

- The files `MySort.py` and `Random.py` are called within the scripts and should be 
downloaded or cloned to run properly


    
    
