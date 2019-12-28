from NeuralNetwork.CoreObject.Node import Node
from NeuralNetwork.CoreObject.GraphComponent import GraphComponent
from NeuralNetwork.CoreObject.Variable import Constant

from .Basic import Multiply, Substract, Power

import numpy as np

from typing import List, Tuple

class Activation:
	def computeShape( self, shape1 )->Tuple[int]:
		return shape1

class Sigmoid( Activation, Node ):

	def __init__( self, x:GraphComponent, **kwargs ):
		super( Sigmoid, self ).__init__( inputs=[x], **kwargs )

	def forward( self, x:np.ndarray )->np.ndarray:
		return 1 / ( 1 + np.power( np.e, -1*x ) )

	def backward( self, dL:GraphComponent )->List[GraphComponent]:
		d = Substract( self, Power( self, 2 ) )
		dX = Multiply( dL, d, name="diff_{}".format(self._input[0].name) )

		return [dX]

class Tanh( Activation, Node ):

	def __init__( self, x:GraphComponent, **kwargs ):
		super( Tanh, self ).__init__( inputs=[x], **kwargs )

	def forward( self, x:np.ndarray )->np.ndarray:
		return np.tanh( x )

	def backward( self, dL:GraphComponent )->List[GraphComponent]:
		d = Substract( Constant((1,), np.array([1])), Power( self, 2 ) )
		dX = Multiply( dL, d,name="diff_{}".format(self._input[0].name) )

		return [dX]

class Relu( Activation, Node ):

#	HACK : For create node that do derivetive of sigmoid
	class __DiffRelu( Activation, Node ):
		def __init__( self, x:GraphComponent, **kwargs ):
			super().__init__( inputs=[x], **kwargs )

		def forward( self, x:np.ndarray )->np.ndarray:

			dX = x > 0
			return dX.astype( np.float64 )

	def __init__( self, x:GraphComponent, **kwargs ):
		super( Relu, self ).__init__( inputs=[x], **kwargs )

	def forward( self, x:np.ndarray )->np.ndarray:
		return np.maximum( x, 0 )

	def backward( self, dL:GraphComponent )->List[GraphComponent]:
		dx = Multiply( dL, self.__DiffRelu( self._input[0] ), name = "diff_{}".format(self._input[0].name) )

		return [dx]

class SoftMax( Activation, Node ):

#	HACK : For create node that do derivetive of sigmoid
	class __DiffSoftmax( Activation, Node ):
		def __init__( self, x:GraphComponent, **kwargs ):
			super().__init__( inputs=[x], **kwargs )

		def forward( self, x:np.ndarray )->np.ndarray:
			print( x.reshape(-1, 2, 1), x  )
			print( "**", np.matmul( x.reshape(-1, 2, 1), np.transpose(x.reshape(-1, 2, 1), axes = (0, 2, 1) ) ) )
			dx =  np.matmul( x.reshape(-1, 2, 1), np.transpose(x.reshape(-1, 2, 1), axes = (0, 2, 1) ) )
			dx = x - dx.sum( axis = 1 )
			# assert False
			return dx

	def __init__( self, x:GraphComponent, **kwargs ):
		super( SoftMax, self ).__init__( inputs=[x], **kwargs )

	def forward( self, x:np.ndarray )->np.ndarray:
		exp = np.exp(x - np.amax(x, axis=0 ))
		# print( (x - np.amax(x, axis=0 ))[:,:2], x.shape )
		# print( x[:,:2], np.amax(x, axis=0 )[:2] )
#	HACK : FIX DIM
		softMax = exp / np.sum( exp, axis = 0, keepdims=True )
		return softMax

	def backward( self, dL:GraphComponent )->List[GraphComponent]:
		dx = Multiply( dL, self.__DiffSoftmax( self ), name = "diff_{}".format(self._input[0].name) )

		return [dx]