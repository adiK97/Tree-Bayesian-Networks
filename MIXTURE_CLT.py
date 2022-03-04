from __future__ import print_function
import numpy as np
from Util import *
import sys
import time
import random
from sklearn.utils import resample
from CLT_class import CLT
 

class MIXTURE_CLT():
   
    def __init__(self):
        self.n_components = 0 # number of components
        self.mixture_probs = None # mixture probabilities
        self.clt_list =[]   # List of Tree Bayesian networks
    '''
        Learn Mixtures of Trees using the EM algorithm.
    '''
    def learn(self, dataset, n_components=2, max_iter=50, epsilon=format(1e-1, '.5f'), r = None):
        # For each component and each data point, we have a weight
        weights=np.ones((n_components , dataset.shape[0]))
        self.mixture_probs = np.ones(n_components)
        for i in range(n_components):
            self.clt_list.append(CLT())
            self.clt_list[i].learn(dataset)
        # Randomly initialize the chow-liu trees and the mixture probabilities
        # Your code for random initialization goes here
        ## Implementing Random Forest in this code as well. If the parameter 'r' is passed while calling this function then, this will run for Q3 ie Random Forest
        if r:
            for i in range(n_components):
                tempdata = resample(dataset)
                self.clt_list[i].learn(tempdata)
                self.clt_list[i].update(tempdata, weights[i], r)
            like = self.computeLL(dataset , n_components , r) / dataset.shape[0]
            print("Likelihood - ", like)


        else:
            a = np.random.rand(n_components)
            for i in range(len(a)):
                self.mixture_probs[i] = a[i] / np.sum(a)  

            like = 0.0
            old_like = 999
            prob = np.zeros((n_components , dataset.shape[0]))

            for itr in range(max_iter):     
                print("Iteration - ",itr)
                print("N components - ",n_components)
                #E-step: Complete the dataset to yield a weighted dataset
                # We store the weights in an array weights[ncomponents,number of points]
                #Your code for E-step here
                if abs(old_like - like) <= epsilon:
                    break
                
                for i in range(n_components):
                    for j in range(dataset.shape[0]):
                        prob[i][j] = self.clt_list[i].getProb(dataset[j])
                    weights[i] = np.multiply(self.mixture_probs[i], prob[i])/np.sum(np.multiply(self.mixture_probs[i], prob[i]))
                    # print(weights.shape)
            
                # # M-step: Update the Chow-Liu Trees and the mixture probabilities
                # #Your code for M-Step here
                for i in range(n_components):
                    self.clt_list[i].update(dataset, weights[i])
                    self.mixture_probs[i] = weights[i].sum()/len(weights[i])
                    
                old_like = like
                like = self.computeLL(dataset , n_components) / dataset.shape[0]
                print(like)

    # """
    #     Compute the log-likelihood score of the dataset
    # """
    def computeLL(self, dataset , n_components , r = None):
        ll = 0.0
        likelihood=0.0

        # Write your code below to compute likelihood of data
        #   Hint:   Likelihood of a data point "x" is sum_{c} P(c) T(x|c)
        #           where P(c) is mixture_prob of cth component and T(x|c) is the probability w.r.t. chow-liu tree at c
        #           To compute T(x|c) you can use the function given in class CLT
        if not r:
            for i in range (dataset.shape[0]):
                for j in range(n_components):
                    likelihood=likelihood + self.mixture_probs[j]*self.clt_list[j].getProb(dataset[i])
                ll=ll+ np.log(likelihood)
        else:
            for k in range(n_components):		
                ll += self.clt_list[k].computeLL(dataset)
								
        return ll
   
   