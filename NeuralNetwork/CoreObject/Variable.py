from .GraphComponent import GraphComponent
from .Graph import _default_graph

import numpy as np

class Variable( GraphComponent ):

	def __init__( self, **kwargs ):
		super( Variable, self ).__init__( **kwargs )

		self._value: np.ndarray = None
		self._output: np.ndarray = None

	def setValue( self, output:np.ndarray ):
		raise NotImplementedError

	def setOutput( self ):
		self._output = self._value.copy()

	def getOutput( self )->np.ndarray:
		assert self._output is not None, "This Variable {} isn't set value".format( self._name )
		return super( Variable, self ).getOutput()

class Constant( Variable ):

	def __init__( self, value:np.ndarray, **kwargs ):
		super( Constant, self ).__init__( **kwargs )
		self._value = value

		_default_graph.constant.append( self )

	def setValue( self, output:np.ndarray ):
		assert False, "Cannot set value for Constant type"

class PlaceHolder( Variable ):

	def __init__( self, **kwargs ):
		super( PlaceHolder, self ).__init__( **kwargs )

		_default_graph.placeHolder.append( self )

	def setValue( self, output:np.ndarray ):
		self._value = output

class Trainable( GraphComponent ):

	def __init__( self, initialValue: np.ndarray, **kwargs ):
		super( Trainable, self ).__init__( **kwargs )

		self._value = initialValue

		_default_graph.trainable.append( self )

	def setValue( self, output:np.ndarray ):
		self._value = output