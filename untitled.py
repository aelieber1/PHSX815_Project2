""" 
Goal Data Analysis Code

Purpose: To analyze the data and perform a hypothesis test following the procedure below:
    1. From the command line, the user can input the two datasets for the two hypotheses we are testing.
    2. The code will read in each file, and create an array of the measurements.
    
Author: @aelieber1
Code also adapted from these sources:
    - 
    
    
    
    FIX ME FIX ME !!!!!!!!!!!!!&&&&&&&&&&!!!!!!!!!!!!!&&&&&&&&&&&&&
    
""" 

# Import necessary packages
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import poisson, gamma

# import system, Random class, and MySort class
sys.path.append(".")
from MySort import MySort
from Random import Random

# Main function for our analysis code
if __name__ == "__main__":
   
    haveInputH0 = False
    haveInputH1 = False
    
    # To pass in the data file to be analyzed from the command line
    if '-inputH0' in sys.argv:
        p = sys.argv.index('-inputH0')
        InputFileH0 = sys.argv[p+1]
        haveInputH0 = True
            
    if '-inputH1' in sys.argv:   
        p = sys.argv.index('-inputH1')
        InputFileH1 = sys.argv[p+1]
        haveInputH1 = True
        
    if '-h' in sys.argv or '--help' in sys.argv or not haveInputH0 or not haveInputH1:
        print ("Usage: %s [options] [input file]" % sys.argv[0])
        print ("  options:")
        print ("   --help(-h)          print options")
        print (" -inputH0 (filename)   first hypothesis data to be analyzed")
        print (" -inputH1 [filename]   alternative hypothesis data to be analyzed")
        sys.exit(1)
    
        
    """ 
    Read in data and computer numerical estimate of probability distribution
    """
    Nmeas = 1
    
    # Read in and sort Data from Hypothesis H0
    Nmeas0 = 1
    H0goals = []
    Nexp0 = 0
    
    with open(InputFileH0) as ifile:
        for line in ifile:
            Nexp0 += 1
            lineVals = line.split()
            Nmeas0 = len(lineVals)
            for v in lineVals:
                H0goals.append(float(v))
    
    Sorter = MySort()
    H0goals = Sorter.BubbleSort(H0goals)
    H0_max = np.max(H0goals)
    H0_min = np.min(H0goals)
    H0_bins = list(range(int(H0_max + 1.0))) # add one, so that the max number of goals will be included in bin count.     

                
    # Read in and sort Data from Hypothesis H1
    Nmeas1 = 1
    H1goals = []
    
    with open(InputFileH1) as ifile:
        for line in ifile:
            lineVals = line.split()
            Nmeas1 = len(lineVals)
            for v in lineVals:
                H1goals.append(float(v))
    
    H1goals = Sorter.BubbleSort(H1goals)
    H1_max = np.max(H1goals)
    H1_min = np.min(H1goals)
    H1_bins = list(range(int(H1_max + 1.0))) 
    # add one, so that the max number of goals will be included in bin count - ith bin = number of times i goals were scored
    
    # For our simulation, the number of measurments should match, so this step checks that  
    if Nmeas1 == Nmeas0:
        Nmeas = Nmeas1

        
    """ Histogram of Vector of Outcomes """
    
    title1 = "H0 Simulated Data - Alpha: 2 & Beta: 1.5"
    title2 = "H1 Simulated Data - Alpha: 4 & Beta: 2.5"
    
    plt.hist(H0goals, bins=H0_bins, alpha=0.5, histtype="bar", color='purple', edgecolor="black", label=title1)
    plt.hist(H1goals, bins=H1_bins, alpha=0.5, histtype="bar",color='green', edgecolor="black", label=title2)
    plt.title("Histogram of Simulated Data")
    plt.legend()
    plt.xlabel("Number of Goals Scored in Single Game")
    plt.ylabel("Number of Measurements")
    plt.show()
    
    
    """ Probability Distribution Estimate """
    # Number of total experiments
    Nmeas = Nmeas0 * Nexp0
    
    #create a list of the probabilities for H0
    H0_probs = []
    H0_counts = []

    for i in H0_bins:
        counts = H0goals.count(i)
        prob = counts / Nmeas
        H0_probs.append(prob)
        H0_counts.append(counts)
    
    #print("H0 num of goals: ", H0_bins)
    #print("H0 probs: ", H0_probs)
    #print("H0 counts: ", H0_counts)
    
    # create a list of the probabilities or likelihoods for H1
    H1_probs = []
    H1_counts = []
    
    for l in H1_bins:
        count = H1goals.count(l)
        probs = count/ Nmeas
        H1_probs.append(probs)
        H1_counts.append(count)
    
    #print("H1 num of goals: ", H1_bins)
    #print("H1 probs: ", H1_probs)
    #print("H1 counts: ", H1_counts)
    
    """At the end of this step, we have three arrays at our disposal (for each hypothesis) One array is the number of goals scored in the dataset, nexxt is the probabilities of scoring that number of goals in a game, and lastly, the counts"""
    
    """ Histogram of Probability Distribution """
    title1 = "H0 Probabilities - Alpha: 2 & Beta: 1.5"
    title2 = "H1 Probabilities - Alpha: 4 & Beta: 2.5"
    
    plt.plot(H0_bins, H0_probs, color='purple', label=title1)
    plt.plot(H1_bins, H1_probs, color='green', label=title2)
    plt.title("Probability Distribution")
    plt.legend()
    plt.ylabel("Probability")
    plt.xlabel("Number of Goals Scored")
    plt.show()
    
    
    """ Calculate the Log Likelihood Ratio """
    # In this portion of the code, we will actually calculate the Log Likelihood Ratio (LLR) for the data. We will find the LLR for each experiment we did. 
    
    # LLR for Hypothesis H0
    LLR_H0 = []
    with open(InputFileH0) as ifile:
        H0_data = np.loadtxt(InputFileH0, dtype=float)
        print("H0 Data: ", H0_data)
        
        # loop over each experiment
        for i in H0_data:
            s = 2
    
    # LLR for Hypothesis H1
    H1_data = np.loadtxt(InputFileH1, dtype=int, skiprows=1)
    print("H1 Data: ", H1_data)
    
    
    