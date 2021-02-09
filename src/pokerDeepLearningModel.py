#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 21:33:07 2018

@author: brandinho
"""

from pathlib import Path
import random
import numpy as np
import tensorflow.compat.v1 as tf
from pokerDeck import pokerDeck
from pokerProbabilities import fetchProbabilityArray, statusDictToInputArray, simulateProbability
from pokerCombinatorics import findHandStatus
from pokerNeuralNetwork import probabilityApproximator
from pokerDeepLearningModelLib import mock_array
from pokerDeepLearningModelLib import simulationDeck
from pokerDeepLearningModelLib import evaluationDeck
from pokerDeepLearningModelLib import tableEvaluationDeck
import pokerDeepLearningModelLib


tf.disable_eager_execution()



### Variables to toggle ###

num_data_points = 100            # The number of instances we want in our sample dataset
use_existing_model = True        # Do you want to use a pre-computed model or train a new one?
inference_sample_size = 100      # Size of sample to test the model on



### Creating the different instances in our dataset ###

sampleDeck = pokerDeck()         # We use the pokerDeck class to draw cards from
sampleHands = []                 # We make an empty list to append the hole cards to
sampleTableCards = []            # We make an empty list to append the board cards to   
tableCardsOptions = [0, 3, 4, 5] # We will randomly select from this list to determine how many cards appear on the table     

for i in range(num_data_points):
    sampleDeck.shuffleDeck()
    
    sampleHands.append(sampleDeck.currentDeck[:2])
    cleanDeck = sampleDeck.currentDeck[2:] # After we take the two hole cards we remove them from the deck
    
    sampleTableCards.append(cleanDeck[:random.choice(tableCardsOptions)])

### We initialize empty arrays for the inputs and the labels ###

probabilityInputList = np.zeros((len(sampleHands), len(mock_array)))
probabilityList = np.zeros((len(sampleHands),2))



### We run through the loop to generate the inputs and labels to use during training ###

for j in range(len(sampleHands)):

    ### We print which iteration we are on every 100 instances to keep track of our status ###
    if j % 100 == 0:
        print('You are at iteration {}'.format(j))
    print('sampleHand: %s' % sampleHands[j])
    print('sampleTable: %s' % sampleTableCards[j])

    
    evaluationDeck.shuffleDeck(); evaluationDeck.table = sampleTableCards[j]

        
    ### Below we combine all of the different states into one input vector (which is assigned to the input array) ###
    
    probabilityInputList[j,] = pokerDeepLearningModelLib.getProbabilityInputList(sampleHands[j], sampleTableCards[j])
    
    ### Below we simulate the probability to use as our label ###
    
    temp_prob_array = simulateProbability(sampleHands[j], sampleTableCards[j], simulationDeck, 1000)
    probabilityList[j,] = temp_prob_array[-1,]


"""
We initialize the tensorflow graph for our model

We have the option to load the pre-computed neural network or to train a new model from scratch
"""

tf.reset_default_graph()
sess = tf.Session()


if use_existing_model:
    my_probability_file = Path("Probability Model/checkpoint")
    if my_probability_file.is_file() == True:
        probability_saver = tf.train.import_meta_graph("Probability Model/ProbabilityApproximator.meta")
        probability_saver.restore(sess, tf.train.latest_checkpoint("Probability Model"))
        
        graph = tf.get_default_graph()
        probabilityFunction = probabilityApproximator(sess, probabilityInputList.shape[1], 0.0005, use_existing_model, graph)
    else:
        raise ValueError("You do not have an existing model, please change this variable to 'False'")
else:
    graph = None
    probabilityFunction = probabilityApproximator(sess, probabilityInputList.shape[1], 0.0005, use_existing_model, graph)
    sess.run(tf.global_variables_initializer())

### We split the data into a training and testing set ###

n_training_set = 90
train_X, test_X = probabilityInputList[:n_training_set,], probabilityInputList[n_training_set:,]
train_Y, test_Y = probabilityList[:n_training_set,], probabilityList[n_training_set:,]

print("train_X: %s" % train_X)

if use_existing_model == False:    
    
    ### Perform Training ###
    
    training_error_array, testing_error_array = probabilityFunction.trainModel(train_X, train_Y, 10000, 25, test_X, test_Y)


### Perform Inference ###

test_predictions = sess.run(probabilityFunction.approximate_probability, {probabilityFunction.inputs: test_X[:inference_sample_size,]})
print('test_predictions: %s' % test_predictions)
test_labels = test_Y[:inference_sample_size,]
test_MAE = np.mean(abs(test_predictions - test_labels), axis = 0)

print("\nMAE for the sample \nP(Win): {:.2f}% \nP(Tie): {:.2f}%".format(test_MAE[0]*100, test_MAE[1]*100))
