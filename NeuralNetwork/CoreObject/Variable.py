from .GraphComponent import GraphComponent
from .Graph import _default_graph
from .utility import compareShape

from typing import Tuple

import numpy as np

class Variable( GraphComponent ):

	def __init__( self, shape: Tuple, **kwargs ):
		super( Variable, self ).__init__( **kwargs )

		self._value: np.ndarray = None
		self._output: np.ndarray = None

		self._shape = shape

	def setValue( self, output:np.ndarray ):
		raise NotImplementedError

	def setOutput( self ):
		self._output = self._value.copy()

	def getOutput( self )->np.ndarray:
		assert self._output is not None, "This Variable {} isn't set value".format( self._name )
		return super( Variable, self ).getOutput()

	@property
	def shape( self ):
		return self._shape

class Constant( Variable ):

	def __init__( self, shape, value:np.ndarray, **kwargs ):
		super( Constant, self ).__init__( shape, **kwargs )
		assert compareShape(value.shape, shape)
		self._value = value

		_default_graph.constant.append( self )

	def setValue( self, output:np.ndarray ):
		assert False, "Cannot set value for Constant type"

class PlaceHolder( Variable ):

	def __init__( self, shape, **kwargs ):
		super( PlaceHolder, self ).__init__( shape, **kwargs )

		_default_graph.placeHolder.append( self )

	def setValue( self, output:np.ndarray ):
		assert compareShape(output.shape, self._shape)
		self._value = output

class Trainable( Variable ):

	def __init__( self, shape, initialValue: np.ndarray, **kwargs ):
		super( Trainable, self ).__init__( shape, **kwargs )

		self._value = initialValue

		assert initialValue.shape == shape

		_default_graph.trainable.append( self )

	def setValue( self, output:np.ndarray ):
		assert compareShape(output.shape, self._shape)
		self._value = output