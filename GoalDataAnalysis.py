""" 
Goal Data Analysis Code

Purpose: To analyze the data and perform a hypothesis test following the procedure below:
    1. From the command line, the user can input the two datasets for the two hypotheses we are testing.
    2. The code will read in each file, and create an array of the measurements.
    
* You will notice several commented out print statements. Those are mainly used to track that inputs, outputs, and calculations are being done correctly for debugging purposes. If you are are running into an issue, those can help find the root cause. 
    
Author: @aelieber1
Code also adapted from these sources:
    - @crogan & associated PHSX 815 homework assignments
    - statistics by Jim - https://statisticsbyjim.com/probability/gamma-distribution/
    - https://vioshyvo.github.io/Bayesian_inference/Bayesian-inference-2017.pdf
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

#Defining function to check for common elements in two lists
# Help from this Tutorial: https://www.tutorialspoint.com/python-check-if-two-lists-have-any-element-in-common
def commonelems(x,y):
    common=0
    for value in x:
        if value in y:
            common=1
    if(not common):
        print("The LLRs have no common elements")
        return False
    else:
        print("The LLRs have common elements")
        return True

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
    H0_bins = list(range(int(H0_max + 1.0))) 
    # added one, so that the max number of goals will be included in bin count.    
    # We define the bins in this way so that the histogram can be binned by integer inputs and puts things in the proper format to numerically compute the probability 

                
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
    
    title1 = "H0 Simulated Data"
    title2 = "H1 Simulated Data"
    
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
    title1 = "H0 Probabilities"
    title2 = "H1 Probabilities"
    
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
    LogLikeRatio_H0 = []
    with open(InputFileH0) as ifile:
        H0_data = np.loadtxt(InputFileH0, dtype=int)
        #print("H0 Data: ", H0_data)
        
        # loop over each experiment
        for i in H0_data:
            result_H0 = 0
            log_H0 = 1
            
            for j in i:
                count_H0 = H0goals.count(j)
                prob_H0 = count_H0 / Nmeas
                
                count_H1 = H1goals.count(j)
                prob_H1 = count_H1 / Nmeas
                
                # Band-aid to ensure functionality so that calculations do not become undefined 
                if prob_H0 == 0.000:
                    prob_H0 = 1 / Nmeas
                if prob_H1 == 0.000:
                    prob_H1 = 1 / Nmeas
                
                log_H0 = math.log(prob_H1 / prob_H0)
                result_H0 = result_H0 + log_H0
            
            LogLikeRatio_H0.append(result_H0)
        print("LogLikeRatio for H0: ", LogLikeRatio_H0)
        print("length: ", len(LogLikeRatio_H0))
            
    
    # LLR for Hypothesis H1
    LogLikeRatio_H1 = []
    with open(InputFileH1) as ifile:
        H1_data = np.loadtxt(InputFileH1, dtype=float)
        #print("H1_data: ", H1_data)
        
        # loop over each experiment
        for i in H1_data:
            result_H1 = 0
            log_H1 = 1
            
            for j in i:
                count_H0 = H0goals.count(j)
                prob_H0 = count_H0 / Nmeas
                #print(prob_H0)
            
                count_H1 = H1goals.count(j)
                prob_H1 = count_H1 / Nmeas
                #print(prob_H1)
                
                # Band-aid to ensure functionality 
                if prob_H0 == 0.000:
                    prob_H0 = 1 / Nmeas
                if prob_H1 == 0.000:
                    prob_H1 = 1 / Nmea
                        
                log_H1 = math.log(prob_H1 / prob_H0)
                result_H1 = result_H1 + log_H1
                
            LogLikeRatio_H1.append(result_H1)
                
        print("LogLikeRatio for H1: ", LogLikeRatio_H1)
        print("length: ", len(LogLikeRatio_H1))
    
    # Calculate the critical test statistic of H0
    # Set confidence level
    alpha = 0.92
    
    # Find critical lambda value that corresponds to that significance level
    lambda_crit = np.quantile(LogLikeRatio_H0, alpha)
    print("Lambda crit: ", lambda_crit)
    
    # Find Hypothesis Test Beta and Power of the Test (1-Beta)
    # *note this is different than the previous beta for the gamma function*
    difference_array = np.absolute(LogLikeRatio_H1 - lambda_crit)
    index = difference_array.argmin()
    print(index)
    
    # Number of experiments 
    Nexp = len(LogLikeRatio_H1)
    
    # Calculate Beta
    beta = LogLikeRatio_H1[index] / Nexp
    # Checking that Beta is within value
    if beta in range(0, 1):
        print("Beta: ", beta)
    else: 
        print(" Please check plots to ensure data is suitable for hypothesis testing, a.k.a. do they overlap at all?")
        
    # Calculate the Power of the test
    power_of_test = 1 - beta
    print("Power of Test: ", power_of_test)

    """Plot the Log Likelihood Ratios for these Hypotheses"""
    n, bins, patches = plt.hist(LogLikeRatio_H0, bins='auto', density=True, 
                                histtype='stepfilled', color='lightblue', ec='darkcyan', 
                                orientation='vertical', alpha=0.75, 
                                label='Probability(X|H0)')
    n, bins, patches = plt.hist(LogLikeRatio_H1, bins='auto', density=True, 
                                histtype='stepfilled', color='bisque', ec='orange', 
                                orientation='vertical', alpha=0.75, 
                                label='Probability(X|H1)')
   
    
    # Plot Formatting
    title = ' Simulated Random Sampling of Goals under Rate Prior Hypotheses'
    plt.xlabel('$\\lambda = \\log({\\cal L}_{\\mathbb{H}_{1}}/{\\cal L}_{\\mathbb{H}_{0}})$', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True)
    
    # plot critical lambda
    plt.axvline(x = lambda_crit, color = 'palevioletred', linewidth = 1, 
                label = 'Critical Test Statistic for α = ' + str(alpha) + ' Confidence Level')
    
    # display the beta / power of test calculated
    plt.text(-10, 0.15, 'Power of the Test $1-\u03B2$: ' + str(power_of_test), fontsize = 10)
        
    plt.legend()
    plt.show()