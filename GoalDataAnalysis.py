"""
Goal Data Analysis Code 

Purpose: To analyze the data and perform a simple hypothesis test following the steps outlined below.
    1. In command line, input the two datasets (simulated code under different hypotheses, that need to be tested
    2. Read in each file, read in the rate, and create an array with all of the measurements - an additional array 
        is made of the averages per experiment as well. 
    3. For these data, each array of observations is sorted, quantiles are calculated, and the data is plotted as 
        histograms to visualize the data before further analysis is completed. 
    4. Computed the LogLikelihood Ratio for each hypothesis dataset according to Poisson probability
    5. From this, the critical test statistic is calculated according to a specific alpha value
    6. Finally this data is plotted, along with the critical test statistic, and the power of the test is displayed

Author: @aelieber1
Code Adapted from these sources: 
    - @crogan PHSX815 Github Week 1 & 2
    - Documentation for NumPy Arrays
    - Documentation for Scipy.Stats Poisson - Probability Mass Function
    - Documentation for Matplotlib.pyplot - for plotting formatting
    - Geeks for Geeks Website to aid with language of 
            - array manipluation  
            - formatting text on plot
            - index selection of NumPy arrays
    - Utlized some StackOverflow forums to circumvent issues in coding outoput/structure
"""

# Import Necessary Packages
import sys
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import poisson

# import our Random class from Random.py file
sys.path.append(".")
from MySort import MySort

# main function for our CookieAnalysis Python code
if __name__ == "__main__":
   
    haveInput = False
    haveInput2 = False
    
    # To pass in the data file to be analyzed from the command line
    if '-inputH0' in sys.argv:
        p = sys.argv.index('-inputH0')
        InputFile = sys.argv[p+1]
        haveInput = True
            
    if '-inputH1' in sys.argv:   
        p = sys.argv.index('-inputH1')
        InputFile2 = sys.argv[p+1]
        haveInput2 = True
    
    
    if '-h' in sys.argv or '--help' in sys.argv or not haveInput or not haveInput2:
        print ("Usage: %s [options] [input file]" % sys.argv[0])
        print ("  options:")
        print ("   --help(-h)          print options")
        print (" -inputH0 (filename)   first hypothesis data to be analyzed")
        print (" -inputH1 [filename]   alternative hypothesis data to be analyzed")
        sys.exit(1)
    
    """ 
    Read in data: 
    (1) Get Rate used, 
    (2) Append number of goals measured to an array, 
    (3) Calculate the average per experiment, average of goals per set 
    """
    # Reading in Data from Hypothesis 0
    Nmeas = 1
    goals = []
    goals_avg = []
    need_rate = True
    
    with open(InputFile) as ifile:
        for line in ifile:
            if need_rate:
                need_rate = False
                rate = float(line)
                continue
            
            lineVals = line.split()
            Nmeas = len(lineVals)
            t_avg = 0
            for v in lineVals:
                t_avg += float(v)
                goals.append(float(v))

            t_avg /= Nmeas
            goals_avg.append(t_avg)\
    
    #print("goals 1: ", goals)
    
    # Reading in Data from Hypothesis 1
    Nmeas2 = 1 
    goals2 = []
    goals2_avg = []
    need_rate2 = True
    
    with open(InputFile2) as ifile:
        for line in ifile:
            if need_rate2:
                need_rate2 = False
                rate2 = float(line)
                continue
                
            lineVals2 = line.split()
            Nmeas2 = len(lineVals2)
            t_avg2 = 0
            for m in lineVals2:
                t_avg2 += float(m)
                goals2.append(float(m))
            
            t_avg2 /= Nmeas2
            goals2_avg.append(t_avg)\
            
    #print("goals 2: ", goals2)
    
    """ Analysis of Data """
    
    """ Sort the Arrays """
    Sorter = MySort()
    
    goals = Sorter.BubbleSort(goals)
    goals_avg = Sorter.BubbleSort(goals_avg)
    
    goals2 = Sorter.BubbleSort(goals2)
    goals2_avg = Sorter.BubbleSort(goals2_avg)
    
    
    """ Code to calculate different quantimes of the assorted arrays of outcomes """
    #print("times: ", times)
    #print("times_avg: ", times_avg)
    
    # Quantiles of Times array
    print("goals of H0 : ", goals)
    print("Q1 quantile of goals H0: ", np.quantile(goals, .25))
    print("Q2 quantile of goals H0: ", np.quantile(goals, .50))
    print("Q3 quantile of goals H0: ", np.quantile(goals, .75))

    print("goals of H1 : ", goals2)
    print("Q1 quantile of goals H1: ", np.quantile(goals2, .25))
    print("Q2 quantile of goals H1: ", np.quantile(goals2, .50))
    print("Q3 quantile of goals H1: ", np.quantile(goals2, .75))
    
    
    """ Plotting Data to Visualize """
    
    # create histogram of the times (or measurement data)
    n, bins, patches = plt.hist(goals, bins=18, density=True, histtype='stepfilled', color='lightblue', ec='darkcyan', orientation='vertical', alpha=0.75, label='H0 Data with Rate: '+str(rate))
    n, bins, patches = plt.hist(goals2, bins=18, density=True, histtype='stepfilled', color='bisque', ec='orange', orientation='vertical', alpha=0.75, label='H1 Data with Rate: '+str(rate2))

    
    # plot formating options
    title = ' Simulated Random Sampling of Goals under Two Different Rate Parameter Hypotheses'
    plt.xlabel('Average Number of Goals', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True)
    
    # add vertical lines to show the quantiles:
    plt.axvline(x = np.quantile(goals, .25), color = 'darkgreen', linewidth = 1.5, label = 'Quantile 1 of H0')
    plt.axvline(x = np.quantile(goals, .50), color = 'royalblue', linewidth = 1.5, label = 'Quantile 2 or Median of H0')
    plt.axvline(x = np.quantile(goals, .75), color = 'mediumpurple', linewidth = 1.5, label = 'Quantile 3 of H0')
    
    plt.axvline(x = np.quantile(goals2, .25), color = 'maroon', linewidth = 1.5, label = 'Quantile 1 of H1')
    plt.axvline(x = np.quantile(goals2, .50), color = 'lightcoral', linewidth = 1.5, label = 'Quantile 2 or Median of H1')
    plt.axvline(x = np.quantile(goals2, .75), color = 'gold', linewidth = 1.5, label = 'Quantile 3 of H1')
    
    # Show Legend
    plt.legend()

    # show figure (program only continue once closed)
    plt.show()
    
    
    """ Calculating the Log Likelihood Ratio for Each Experiment Under Each Hypothesis """
    
    """ Algorithm to Calculate LogLikeRatio for H1 """
    LogLikeRatio_H0 = []
    with open(InputFile) as ifile:
        H0_data = np.loadtxt(InputFile, dtype=int, skiprows=1)
        #print(H0_data)
        
        # loops over each experiment
        for i in H0_data:        
            result_H0 = 1
            result_H1 = 1
            LLR_H0 = 1

            # loops over each value for each experiment to calculate the probability
            for j in i:
                prob_H0 = poisson.pmf(k=j, mu=rate)
                prob_H1 = poisson.pmf(k=j, mu=rate2)
                result_H0 = result_H0 * prob_H0
                result_H1 = result_H1 * prob_H1
                
            LLR_H0 = math.log( result_H1 / result_H0 )
            # append that to the array of likelihoods
            LogLikeRatio_H0.append(LLR_H0)
            
    LogLikeRatio_H0 = Sorter.BubbleSort(LogLikeRatio_H0) 
    print("LogLikeRatio for H0: ", LogLikeRatio_H0)
    
    
    """ Same Algrithm for Hypothese H1 """
    LogLikeRatio_H1 = []
    with open(InputFile2) as ifile:
        H1_data = np.loadtxt(InputFile2, dtype=int, skiprows=1)
        #print(H1_data)
        
        # loops over each experiment
        for i in H1_data:
            result_H0 = 1
            result_H1 = 1
            LLR_H1 = 1

            # loops over each value for each experiment to calculate the probability
            for j in i:
                prob_H0 = poisson.pmf(k=j, mu=rate)
                prob_H1 = poisson.pmf(k=j, mu=rate2)
                result_H0 = result_H0 * prob_H0
                result_H1 = result_H1 * prob_H1
                
            LLR_H1 = math.log( result_H1 / result_H0 )
            # append that to the array of likelihoods
            LogLikeRatio_H1.append(LLR_H1)
            
    LogLikeRatio_H1 = Sorter.BubbleSort(LogLikeRatio_H1)
    print("LogLikeRatio_H1: ", LogLikeRatio_H1)
    
    
    """Calculate Critical Test Statistic of H0"""
    # Set confidence level
    alpha = 0.98
    
    # Find which critical lambda value that corresponds to that significance level
    lambda_crit = np.quantile(LogLikeRatio_H0, alpha)
    print("lamda crit: ", lambda_crit)
    
  
    """ Finding Beta and the Power of Test (1-Beta) """
    # Find lamda crit in H1 array - index position
    difference_array = np.absolute(LogLikeRatio_H1 - lambda_crit)
    index = difference_array.argmin()
    print(index)
    
    # Number of experiments 
    Nexp = len(LogLikeRatio_H1)
    
    # Calculate Beta
    beta = index / Nexp
    print("Beta: ", beta)

    # Calculate the Power of the test
    power_of_test = 1 - beta
    print("Power of Test: ", power_of_test)
    
    
    """Plot the Log Likelihood Ratios for these Hypotheses"""
    n, bins, patches = plt.hist(LogLikeRatio_H0, bins=18, density=True, 
                                histtype='stepfilled', color='lightblue', ec='darkcyan', 
                                orientation='vertical', alpha=0.75, label='Probability(X|H0) testing hypothesis $\lambda$ = ' + str(rate))
    n, bins, patches = plt.hist(LogLikeRatio_H1, bins=18, density=True, 
                                histtype='stepfilled', color='bisque', ec='orange', 
                                orientation='vertical', alpha=0.75, label='Probability(X|H1)testing hypothesis $\lambda$ = ' + str(rate2))
    
    # Plot Formatting
    title = ' Simulated Random Sampling of Goals under Two Different Rate Parameter Hypotheses'
    plt.xlabel('$\\lambda = \\log({\\cal L}_{\\mathbb{H}_{1}}/{\\cal L}_{\\mathbb{H}_{0}})$', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True)
    
    # plot critical lambda
    plt.axvline(x = lambda_crit, color = 'palevioletred', linewidth = 1, label = 'Critical Test Statistic for α = ' + str(alpha) + ' Confidence Level')
    
    # display the beta / power of test calculated
    plt.text(-25, 0.095, 'Power of the Test $1-\u03B2$: ' + str(power_of_test), fontsize = 10)
    
    plt.legend()
    plt.show()