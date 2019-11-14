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

def predToInput( pred:np.ndarray ):
	
	index = np.argmax( pred )
	return index + 1

def agentFactory( nnConstruct, chromosome = None ):

	stack = []

	lookup = {'lstm':LSTM, 'mlp':MLP}

	for nnType, args, kwargs in nnConstruct:
		stack.append( lookup[nnType](*args, **kwargs) )

	nn = Sequential( stack )

	if chromosome is not None:
		nn.setWeight( chromosome )

	return nn

def loadConfig( path ):

	config = configparser.ConfigParser( )
	config.read( path )

	index = 1

	nnConstruct = []

	while True:
		if str(index) in config['Network']:
			nnConstruct.append( eval( config['Network'][str(index)] ) )

		else:
			break
		
		index += 1

	gaConfig  = {}

	gaConfig['survivalRatio'] = float(config['GA']['survivalRatio'])
	gaConfig['duplicateRation'] = float(config['GA']['duplicateRation'])
	gaConfig['mutationRate'] = float(config['GA']['mutationRate'])
	gaConfig['mutationAmount'] = float(config['GA']['mutationAmount'])
	gaConfig['seed'] = int(config['GA']['seed'])

	maxIter = int(config['ITER']['iter'])

	return nnConstruct, gaConfig, maxIter

nnConstruct, gaConfig, maxIter = loadConfig( './config_lstm.ini' )

pprint.pprint( nnConstruct )

if os.path.isfile( os.path.join( './modelSnake_lstm', 'backup' ) ):

	backupDict = pickle.load( open(os.path.join( './modelSnake_lstm', 'backup' ), 'rb' ) )

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

	if i % 10 == 0:
		with open( os.path.join( './modelSnake_lstm', 'iter_{}'.format( i ) ), 'wb') as f:
			pickle.dump( pop[:10], f )
		
		with open( os.path.join( './modelSnake_lstm', 'iter_score_{}'.format( i ) ), 'wb') as f:
			pickle.dump( pop[:10], f )

	try:
		print( "Saving.." )
		with open( os.path.join( './modelSnake_lstm', 'backup' ), 'wb') as f:
			pickle.dump( {"iter":i, "pop":pop}, f )
	except KeyboardInterrupt:
		with open( os.path.join( './modelSnake_lstm', 'backup' ), 'wb') as f:
			pickle.dump( {"iter":i, "pop":pop}, f )
	finally:
		with open( os.path.join( './modelSnake_lstm', 'backup' ), 'wb') as f:
			pickle.dump( {"iter":i, "pop":pop}, f )

	print( "HIGHEST: ", max(pop, key = lambda x : x[0] )[0] )
	print( "avg: ", sum( [p[0] for p in pop] ) )