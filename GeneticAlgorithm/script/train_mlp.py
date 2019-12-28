from GeneticAlgorithm.GA import GA
from NeuralNetwork.MLP import MLP
from NeuralNetwork.LSTM import LSTM
from NeuralNetwork.Sequential import Sequential
from SnakeGame.Enviroment import Enviroment
from SnakeGame.Renderer import cv2Renderer

import numpy as np

import pickle
import os, sys, glob

import configparser
import pprint

from utils import loadConfig, agentFactory, predToInput, saveBackup, saveIter

nnConstruct, gaConfig, maxIter = loadConfig( './config_mlp.ini' )

pprint.pprint( nnConstruct )

if os.path.isfile( os.path.join( './modelSnake', 'backup' ) ):

	backupDict = pickle.load( open(os.path.join( './modelSnake', 'backup' ), 'rb' ) )

	it = backupDict['iter']
	pop = backupDict[ 'pop' ]

	pop = [ agentFactory( nnConstruct, chromosome = p[1] ) for p in pop ]

else:
	it = 0
	pop = [ agentFactory( nnConstruct ) for _ in range( 1024 ) ]

renderer = cv2Renderer( 13, 13 )
renderer.setRefreshRate( 0 )

envi = Enviroment( 13, 13, renderer )
ga = GA( pop, envi, predToInput )

for i in range( it, maxIter ):
	print( i, "/", maxIter )
	pop = ga.training( **gaConfig )

	pop = sorted( pop, key=lambda x: x[0], reverse=True)

	if i % 25 == 0:
		saveIter( pop, i, './modelSnake' )

	saveBackup('./modelSnake', pop, i  )

	print( "HIGHEST: ", max(pop, key = lambda x : x[0] )[0] )
	print( "avg: ", sum( [p[0] for p in pop] ) )
