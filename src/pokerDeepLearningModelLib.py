import numpy as np
import tensorflow.compat.v1 as tf
from pathlib import Path

from pokerDeck import pokerDeck

from pokerProbabilities import fetchProbabilityArray, statusDictToInputArray, simulateProbability
from pokerCombinatorics import findHandStatus
from pokerNeuralNetwork import probabilityApproximator

mock_array = ['prob1', 'prob2', 'prob3', 'prob4', 'prob5', 'prob6', 'prob7', 'prob8', 
              'pairStatusHand', 'suitedStatusHand', 'straightGapStatusHand', 'lowCardHand', 'highCardHand',
              'pairStatusTable', 'suitedStatusTable', 'straightGapStatusTable', 'tripleStatusTable', 'TwoPairStatusTable', 
              'FullHouseStatusTable', 'RunnerRunnerStatusTable', 'SingleRunnerStatusTable', 
              'Straight - Table', 'Flush - Table', 'Four of a Kind - Table', 'Straight Flush - Table',
              'preflopOneHot', 'flopOneHot', 'turnOneHot', 'riverOneHot']
              

### We create multiple decks that serve different purposes when creating the input vectors ###

simulationDeck = pokerDeck()
evaluationDeck = pokerDeck()
tableEvaluationDeck = pokerDeck()


def getModel(path, sess, graph, n_features):
    my_probability_file = Path(f"{path}/checkpoint")
    if my_probability_file.is_file() == True:
        probability_saver = tf.train.import_meta_graph(f"{path}/ProbabilityApproximator.meta")
        probability_saver.restore(sess, tf.train.latest_checkpoint(path))
        return probabilityApproximator(sess, n_features, 0.0005, True, graph)
    else:
        raise ValueError("You do not have an existing model, please change this variable to 'False'")



def getProbabilityInputList(hand, board):
    """
    Below we calculate the probability of getting various hand rankings
    
    If we achieved a certain hand, all hands at least as good as it get a 1 assigned
    All other hands get a probability assigned
    """
    global evaluationDeck
    global simulationDeck
    sampleCurrentHand, _ = evaluationDeck.evaluateHand(hand)
    sampleCurrentRanking = simulationDeck.handRanking(sampleCurrentHand)
    probArray = fetchProbabilityArray(hand, board, sampleCurrentRanking)
    
    ### Below we find the state of our hole cards using a combination of variables ###
    
    preflopStatusDict = findHandStatus(hand, [])
    statusArray = statusDictToInputArray(preflopStatusDict, "Hand", hand, None)
    
    ### Below we find the state of the board cards using a combination of variables ###
    
    if len(board) != 0:
        tableStatusDict = findHandStatus(board, [])
    else:
        tableStatusDict = {}
    tableStatusArray = statusDictToInputArray(tableStatusDict, "Table", board, tableEvaluationDeck)
    
    ### Below we create a one-hot encoding of the betting round state ###
    
    if len(board) == 0:
        tableStatus = [1,0,0,0]
    elif len(board) == 3:
        tableStatus = [0,1,0,0]
    elif len(board) == 4:
        tableStatus = [0,0,1,0]
    elif len(board) == 5:
        tableStatus = [0,0,0,1]

    ### Below we combine all of the different states into one input vector (which is assigned to the input array) ###
    
    return np.concatenate((probArray, statusArray, tableStatusArray, tableStatus))