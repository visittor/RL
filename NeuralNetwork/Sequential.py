from .BaseNetwork import BaseNetwork

from typing import List
import numpy as np

class Sequential( BaseNetwork ):

	def __init__( self, nnList: List[BaseNetwork] ):

		self._nnList = nnList

		self._nWeight = 0
		
		for nn in self._nnList:

			self._nWeight += nn.nWeight()

	def initWeight( self ):

		for nn in self._nnList:

			nn.initWeight()

	def getFlattenWeight( self ):

		flatW = []

		for nn in self._nnList:

			flatW.append( nn.getFlattenWeight() )

		return np.hstack( flatW )

	def setWeight( self, flatWeight ):

		index = 0

		for nn in self._nnList:

			nn.setWeight( flatWeight[index:index+nn.nWeight()] )

			index += nn.nWeight()

	def predict( self, x ):

		for nn in self._nnList:

			x = nn.predict( x )

		return x

	def resetState( self ):

		for nn in self._nnList:

			nn.resetState()

	def nWeight( self ):

		return self._nWeight