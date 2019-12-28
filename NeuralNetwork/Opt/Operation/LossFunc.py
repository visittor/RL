from NeuralNetwork.CoreObject.Node import Node
from NeuralNetwork.CoreObject.GraphComponent import GraphComponent
from NeuralNetwork.CoreObject.Variable import Constant

from .Basic import Divide, Multiply, Log, Add, Substract

import numpy as np

from typing import List, Tuple

class CategoricalLogLoss( Node ):

	def __init__( self, pred, gt, **kwargs ):
		super().__init__( inputs=[pred, gt], **kwargs )

	def computeShape( self, shape1, shape2 ):
		assert shape1[1] == shape2[1]
		assert shape1[0] == shape2[0]
		assert len(shape1) == len(shape2) == 2

		return (1, 1)

	def forward( self, pred:np.ndarray, gt:np.ndarray )->np.ndarray:
		l = np.sum( np.multiply( gt, np.log( pred ) ), keepdims=True )
		return (-1 * l) / pred.shape[0]

	def backward( self, dL:GraphComponent )->GraphComponent:
		dPred = Multiply( 
						Constant((1,), np.array([-1])),
						Divide( self._input[1], self._input[0] )
						)

		dGT = Multiply( 
						Constant((1,), np.array([-1])),
						Log( self._input[1] )
						)

		return [ 
				Multiply( dL, dPred, name="diff_{}".format(self._input[0].name) ),
				Multiply( dL, dGT, name="diff_{}".format(self._input[1].name) ),
				]

class LogLoss( Node ):

	def __init__( self, pred, gt, **kwargs ):
		super().__init__( inputs=[pred, gt], **kwargs )

	def computeShape( self, shape1, shape2 ):
		assert shape1[1] == shape2[1] == 1
		assert shape1[0] == shape2[0]
		assert len(shape1) == len(shape2) == 2

		return (1, 1)

	def forward( self, pred:np.ndarray, gt:np.ndarray )->np.ndarray:

		l = -gt*np.log(pred) - (1-gt)*np.log(1-pred)
		print( l)
		return np.sum(l) / l.shape[0]

	def backward( self, dL:GraphComponent )->GraphComponent:

		gt = self._input[1]
		pred = self._input[0]

		one = Constant( (1,), np.array([1]) )
		minusOne = Constant( (1,), np.array([-1]) )
		dPred = Substract( Divide( Substract(one, gt), Substract(one, pred) ), Divide( gt, pred ) )
		dPred = Multiply( dL, dPred )

		dGT = Substract( Log( Substract(one, pred) ), Log(pred) )
		dGT = Multiply( dL, dGT )

		return [dPred, dGT ]