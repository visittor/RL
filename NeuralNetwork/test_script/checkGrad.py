from NeuralNetwork.CoreObject.Variable import PlaceHolder
from NeuralNetwork.Opt.Operation.Gradient import check_gradient

from NeuralNetwork.Opt.Operation.Basic import MatMul
from NeuralNetwork.Opt.Operation.Activation import SoftMax, Tanh, Sigmoid
from NeuralNetwork.Opt.Operation.LossFunc import LogLoss

import numpy as np

def checkMatMul( ):
	x = PlaceHolder( (3,2) )
	y = PlaceHolder( (3,2,3) )

	n = MatMul( x, y )

	result = check_gradient( n, [x,y], [np.random.randn(3,2), np.random.randn(3,2,3)] )

	import pprint

	pprint.pprint( result )

def checkSoftmax( ):
	x = PlaceHolder( (5, 2) )

	n = SoftMax( x )

	result = check_gradient( n, [x], [np.random.randn( *x.shape )] )

	import pprint

	pprint.pprint( result )

def checkLoss():
	x = PlaceHolder( (5,1) )
	y = PlaceHolder( (5,1) )

	n = LogLoss( Sigmoid(x), Sigmoid(y) )
	result = check_gradient( n, [x,y], [np.random.randn( *x.shape ), np.random.randn( *y.shape )] )

	import pprint

	pprint.pprint( result )

if __name__ == '__main__':

	checkLoss()