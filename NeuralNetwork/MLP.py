import numpy as np
from typing import Tuple, List

from .Activation import relu, sigmoid, tanh
from .BaseNetwork import BaseNetwork

class MLP( BaseNetwork ):

	ACTIVATION_MAP = {  
						'relu'      : relu,
						'sigmoid'   : sigmoid,
						'tanh'      : tanh,
					}

	def __init__( self, nIn: int, nOut: int, hidden: Tuple, activation: str = 'relu' ):

		self._nPerceptron = [nIn] + list(hidden) + [nOut]

		self._wShape = []

		self._nWeight = 0

		for i in range( len(self._nPerceptron) - 1 ):

			self._wShape.append( (self._nPerceptron[i+1], self._nPerceptron[i]) )

			self._nWeight += self._nPerceptron[i+1] * self._nPerceptron[i]
			self._nWeight += self._nPerceptron[i+1]

		self._wList = []

		self._activation = self.ACTIVATION_MAP[ activation ]

		self.initWeight()

	def initWeight( self ):

		for shape in self._wShape:

			w = np.random.normal( size = shape )
			b = np.random.normal( size = (shape[0], 1) )

			self._wList.append( (w, b) )

	def getFlattenWeight( self ):

		flatWeight = []

		for w, b in self._wList:

			flatWeight.append( w.flatten( order = 'C' ) )
			flatWeight.append( b.flatten( order = 'C' ) )

		return np.hstack(flatWeight)

	def setWeight( self, flatWeight ):

		assert len( flatWeight ) == self.nWeight()

		index = 0

		self._wList = []

		for nOut, nIn in self._wShape:
			
			w = flatWeight[ index : index + (nOut*nIn) ].reshape( nOut, nIn )
			index += nOut * nIn

			b = flatWeight[ index : index + nOut ].reshape( nOut, 1 )
			index += nOut

			self._wList.append( (w, b) )

		assert index == len( flatWeight )

	def predict( self, x ):
		
		x = x.reshape( -1, 1 ).copy()
		assert x.shape[0] == self._nPerceptron[0]

		for w, b in self._wList:

			x = np.matmul( w, x ) + b
			x = self._activation( x )

		x = x.flatten()
		
		assert x.shape[0] == self._nPerceptron[-1]

		return x

	def nWeight( self ):
		return self._nWeight