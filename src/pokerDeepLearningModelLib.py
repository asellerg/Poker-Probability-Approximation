import numpy as np
import tensorflow.compat.v1 as tf
from pathlib import Path
import time

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


_status_dict_to_input_array = 0.0
_fetch_probability_array = 0.0
_find_hand_status = 0.0


def getModel(path, sess, graph, n_features):
    my_probability_file = Path(f"{path}/checkpoint")
    if my_probability_file.is_file() == True:
        probability_saver = tf.train.import_meta_graph(f"{path}/ProbabilityApproximator.meta")
        probability_saver.restore(sess, tf.train.latest_checkpoint(path))
        return probabilityApproximator(sess, n_features, 0.0005, True, graph)
    else:
        raise ValueError("You do not have an existing model, please change this variable to 'False'")



def getProbabilityInputList(hand, board, sampleCurrentRanking=None, table_rank=None, full_rank=None, holeCardsClass=None):
    """
    Below we calculate the probability of getting various hand rankings
    
    If we achieved a certain hand, all hands at least as good as it get a 1 assigned
    All other hands get a probability assigned
    """
    global evaluationDeck
    global simulationDeck
    global _fetch_probability_array
    global _status_dict_to_input_array
    global _find_hand_status
    if sampleCurrentRanking is None:
        sampleCurrentHand, _ = evaluationDeck.evaluateHand(hand)
        sampleCurrentRanking = simulationDeck.handRanking(sampleCurrentHand)
    start = time.time()
    probArray = fetchProbabilityArray(hand, board, sampleCurrentRanking)
    end = time.time()
    _fetch_probability_array += (end-start)
    # print('fetchProbabilityArray: %f' % _fetch_probability_array)
    
    ### Below we find the state of our hole cards using a combination of variables ###
    start = time.time()
    preflopStatusDict = findHandStatus(hand, [], holeCardsClass=holeCardsClass)
    end = time.time()
    _find_hand_status += (end-start)
    # print('findHandStatus: %f' % _find_hand_status)
    start = time.time()
    statusArray = statusDictToInputArray(preflopStatusDict, "Hand", hand, None)
    end = time.time()
    _status_dict_to_input_array += (end-start)
    # print('statusDictToInputArray Hand: %f' % _status_dict_to_input_array)
    
    
    ### Below we find the state of the board cards using a combination of variables ###
    
    if len(board) != 0:
        start = time.time()
        tableStatusDict = findHandStatus(board, [], table_rank=table_rank)
        end = time.time()
        _find_hand_status += (end-start)
        # print('findHandStatus: %f' % _find_hand_status)
    else:
        tableStatusDict = {}
    start = time.time()
    tableStatusArray = statusDictToInputArray(tableStatusDict, "Table", board, tableEvaluationDeck, table_rank=table_rank)
    end = time.time()
    _status_dict_to_input_array += (end-start)
    # print('statusDictToInputArray Table: %f' % _status_dict_to_input_array)
    
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
