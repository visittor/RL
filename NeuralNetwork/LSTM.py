from .MLP import MLP
from .BaseNetwork import BaseNetwork
from .Activation import tanh

import numpy as np

class LSTM( object ):

	def __init__( self, nIn, nOut, hidden ):

		self._forgotten = MLP( nIn + nOut, nOut, hidden, activation='sigmoid' )
		self._selection = MLP( nIn + nOut, nOut, hidden, activation='sigmoid' )
		self._input = MLP( nIn + nOut, nOut, hidden, activation='tanh' )
		self._output = MLP( nIn + nOut, nOut, hidden, activation='tanh' )

		self._cellState = np.zeros( (nOut, ) )
		self._prediction = np.zeros( (nOut, ) )

		self._nOut = nOut
		self._nIn = nIn

		self._nWeight = self._forgotten.nWeight() * 4

	def initWeight( self ):

		self._forgotten.initWeight()
		self._selection.initWeight()
		self._input.initWeight()
		self._output.initWeight()

		self.resetState( )

	def getFlattenWeight( self ):
		
		flatW = []

		flatW.append( self._forgotten.getFlattenWeight() )
		flatW.append( self._selection.getFlattenWeight() )
		flatW.append( self._input.getFlattenWeight() )
		flatW.append( self._output.getFlattenWeight() )

		return np.hstack( flatW )

	def setWeight( self, flatWeight ):

		assert len( flatWeight ) == self.nWeight()

		index = 0

		self._forgotten.setWeight( flatWeight[index:index+self._forgotten.nWeight()] )
		index += self._forgotten.nWeight()

		self._selection.setWeight( flatWeight[index:index+self._selection.nWeight()] )
		index += self._selection.nWeight()

		self._input.setWeight( flatWeight[index:index+self._input.nWeight()] )
		index += self._input.nWeight()

		self._output.setWeight( flatWeight[index:index+self._output.nWeight()] )
		index += self._output.nWeight()

		self.resetState()

	def predict( self, x ):

		assert x.shape[0] == self._nIn

		x = np.hstack( (x, self._prediction) )

		self._cellState = np.multiply( self._cellState, self._forgotten.predict(x) )

		inputGate = np.multiply( self._selection.predict(x), self._input.predict(x) )

		self._cellState += inputGate
		self._prediction = np.multiply( tanh(self._cellState), self._output.predict(x) )

		assert self._prediction.shape[0] == self._nOut
		return self._prediction.copy()

	def resetState( self ):
		self._cellState = np.zeros( (self._nOut, ) )
		self._prediction = np.zeros( (self._nOut, ) )

	def nWeight( self ):
		return self._nWeight