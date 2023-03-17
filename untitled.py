"""

"""

# Import necessary external packages to use
from scipy.stats import poisson
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import sys

# read in lambda data of rate values
file1 = open("rates1.txt", 'r')
H0_rates = file1.read()

file2 = open("rates2.txt",'r')
H1_rates = file2.read()


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

plt.hist(H1_rates, density=True, bins='auto',histtype="bar", alpha=0.2, color='lightblue',ec='black',label= "Sampled rates from H1 Gamma Distribution")

#plt.hist(H0_rates, density=True, bins='auto',histtype="bar", alpha=0.2, color='pink',ec='black', label = "Sampled Rates from H0 Gamma Distribution")

plt.title("Hypothesis Gamma Distribution with Sampled Rate Values")
plt.legend()
plt.show()