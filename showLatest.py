from NeuralNetwork.LSTM import LSTM
from NeuralNetwork.MLP import MLP
from NeuralNetwork.Sequential import Sequential

from SnakeGame.Enviroment import Enviroment
from SnakeGame.Renderer import cv2Renderer, NoGUIRenderer

from GeneticAlgorithm.GA import GA

import sys, os, glob
import pickle
import random

from utils import agentFactory, loadConfig, predToInput

backupFolder = sys.argv[1]
config = sys.argv[2]

fnList = sorted( glob.glob(backupFolder + 'iter_*'), reverse=True)

fn = fnList[0]

print( fn )

chromosomesList = pickle.load( open( fn, 'rb' ) )

chromosomesList = sorted( chromosomesList, key=lambda x: x[0] )

nnConstruct, gaConfig, maxIter = loadConfig( config )

print( nnConstruct )

agents = [ ( score, agentFactory( nnConstruct, chromosome=chro ) ) for score, chro in chromosomesList ]

renderer = cv2Renderer( 13, 13 )
renderer = NoGUIRenderer( 13, 13 )
renderer.setRefreshRate( 100 )

envi = Enviroment( 13, 13, renderer )

ga = GA( [], envi, predToInput )

random.seed( gaConfig['seed'] )
score = ga.testOneAgent( agents[-1][1], render=True )
print( agents )
print( score )
