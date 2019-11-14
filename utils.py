import configparser
import pickle

import os, sys, glob

import numpy as np 

from NeuralNetwork.LSTM import LSTM
from NeuralNetwork.MLP import MLP
from NeuralNetwork.Sequential import Sequential

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

def saveIter( pop, iter, folder ):

	with open( os.path.join( folder, 'iter_{}'.format( str(iter).zfill( 6 ) ) ), 'wb') as f:
		pickle.dump( pop[:10], f )
	
	with open( os.path.join( folder, 'iter_score_{}'.format( str(iter).zfill( 6 ) ) ) , 'wb') as f:
		pickle.dump( pop[:10], f )

def saveBackup( folder, pop, iter ):

	try:
		print( "Saving.." )
		with open( os.path.join( folder, 'backup' ), 'wb') as f:
			pickle.dump( {"iter":iter, "pop":pop}, f )
	except KeyboardInterrupt:
		with open( os.path.join( folder, 'backup' ), 'wb') as f:
			pickle.dump( {"iter":iter, "pop":pop}, f )
	finally:
		with open( os.path.join( folder, 'backup' ), 'wb') as f:
			pickle.dump( {"iter":iter, "pop":pop}, f )