from typing import List, Dict

from .utility import IDGenerator

import numpy as np

_idGenerator = IDGenerator()

class GraphComponent( object ):

	def __init__( self, name: str = None ):

		self._output: np.ndarray = np.array([])

		self._id = next( _idGenerator )
		self._name = name if name is not None else str( self._id )

	def getOutput( self )->np.ndarray:
		return self._output.copy()
	
	def setOutput( self, output:np.ndarray ):
		self._output = output

	@property
	def name( self ):
		return self._name

	@property
	def Id( self ):
		return self._id

	def __eq__( self, other ):

		if not isinstance( other, GraphComponent ):
			return False
		
		return self.Id == other.Id

	def __ne__( self, other ):
		return not self.__eq__

	def __hash__( self ):
		return self.Id

	def __str__( self ):

		return self.name
	
	def __repr__( self ):

		return self.name