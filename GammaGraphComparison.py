"""
Code to Compare Gamma Distribution Graphs for Both Hypotheses

    - This code could have allowed for user input of the alphas, betas, and rate files, but since I was showing this for just my scenario, I didn't take the time to add those commands in. Nevertheless, I wanted to note that here, that this would be a good thing to add if more scenarios were being tested, to ease the workflow of the code I have created
    
Author: @aelieber1
Date: March 10, 2023
"""

from scipy.stats import poisson
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import sys
from numpy import loadtxt

rates1 = []
H0_data_test = np.loadtxt("testrateH0.txt", dtype=str, skiprows=1)
for h in H0_data_test:
    h = float(h)
    rates1.append(h)

rates2 = []
H1_data_test = np.loadtxt("testrateH1.txt", dtype=str, skiprows=1)
for l in H1_data_test:
    l = float(l)
    rates2.append(l)

#define x-axis values
x = np.linspace (0, 30, 2000) 

#calculate pdf of Gamma distribution for each x-value
H0 = stats.gamma.pdf(x, a=2, scale=1.5)
H1 = stats.gamma.pdf(x, a=4, scale=2.5)

#create plot of Gamma distribution
title1 = "H0 Gamma Distribution - Alpha: 2 & Beta: 1.5"
title2 = "H1 Gamma Distribution - Alpha: 4 & Beta: 2.5"

#plt.plot(x, H0, color='purple', label = title1)
plt.plot(x, H1, color='green', label = title2)
plt.plot(x, H0, color='purple', label = title1)

plt.hist(rates2, density=True, bins='auto',histtype="bar", alpha=0.2, color='lightblue',ec='black',label= "Sampled rates from H1 Gamma Distribution")

plt.hist(rates1, density=True, bins='auto',histtype="bar", alpha=0.2, color='pink',ec='black', label = "Sampled Rates from H0 Gamma Distribution")

plt.title("Gamma Distributions of Two Hypothesized Rate Priors")
plt.xlim(-2,32)
plt.legend()
plt.show()