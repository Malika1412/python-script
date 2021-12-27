import numpy as np
import math


# Input: the lists of prior probabilities, likelihood, and test data
# Output: list of corresponding posterior probabilities
#

def posteriorFunc(priorProb, likhd, data):
    '''

       Student implements the function to calculate posterior probabilities here


	'''
    if len(priorProb) == len(likhd):
        summation = 0
        a = data.count(0)
        b = data.count(1)
        for i in range(len(likhd)):
            post = ((1-likhd[i]) ** a) * (likhd[i] ** b) * priorProb[i]
            summation = summation + post

        alpha = 1/summation

        posProb = []
        for j in range(len(likhd)):
            post = ((1-likhd[j]) ** a) * (likhd[j] ** b) * priorProb[j] * alpha
            posProb.append(post)
        return posProb



# Input the lists of prior probabilites, likhd/likelihood, training data, and one test datapoint
# Output: probability that the test datapoint happens
# Note: this function will call posteriorFunc to calculate the posterior probabilites
def predictionFunc(priorProb, likhd, data, fPoint):
    '''

       Student implements the function to calculate predictive probability here

	'''
    new_posProb = posteriorFunc(priorProb, likhd, data)
    new_sum = 0

    for b in range(len(priorProb)):
        if fPoint == 1:
            a = new_posProb[b] * likhd[b]
        else:
            a = new_posProb[b] * (1 - likhd[b])
        new_sum = new_sum + a
        predictProb = new_sum

    return predictProb