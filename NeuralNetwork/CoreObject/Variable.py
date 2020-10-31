from .GraphComponent import GraphComponent
from .Graph import _default_graph
from .utility import compareShape

from typing import Tuple

from abc import ABCMeta, abstractmethod

import numpy as np

class Variable( GraphComponent, metaclass = ABCMeta ):

	def __init__( self, shape: Tuple, **kwargs ):
		super( Variable, self ).__init__( **kwargs )

		self._value: np.ndarray = None
		self._output: np.ndarray = None

		self._shape = shape

	@abstractmethod
	def setValue( self, output:np.ndarray ):
		raise NotImplementedError

	def setOutput( self ):
		self._output = self._value.copy()

	def getOutput( self )->np.ndarray:
		assert self._output is not None, "This Variable {} isn't set value".format( self._name )
		return super( Variable, self ).getOutput()

	def setShape( self, shape:Tuple[int] ):
		self._shape = shape

	@property
	def shape( self ):
		return self._shape

class Constant( Variable ):
	__INSTANCE = {}
	'''
		Value of the constance will be set by __init__ and cannot be
		changed after that.
	'''
	#	This class will be singleton.

	def __new__( cls, shape, value:np.ndarray, **kwargs ):

		if (shape, value.tobytes()) in cls.__INSTANCE:
			return cls.__INSTANCE[ (shape, value.tobytes()) ]

		instance = super( Constant, cls ).__new__( cls )
		instance.__init__( shape, value, **kwargs )
		cls.__INSTANCE[ (shape, value.tobytes()) ] = instance

		#	By returning a instance, python always all __init__
		#	we have to fix this since we don't have to call it again.
		return instance

	def __init__( self, shape, value:np.ndarray, **kwargs ):
		super( Constant, self ).__init__( shape, **kwargs )
		assert compareShape(value.shape, shape)
		self._value = value

		_default_graph.constant.append( self )

	def setValue( self, output:np.ndarray ):
		assert False, "Cannot set value for Constant type"

class PlaceHolder( Variable ):
	'''
		Value of PlaceHolder will be set during session.run
	'''
	def __init__( self, shape, **kwargs ):
		super( PlaceHolder, self ).__init__( shape, **kwargs )

		_default_graph.placeHolder.append( self )

	def setValue( self, output:np.ndarray ):
		assert compareShape(output.shape, self._shape)
		self._value = output

class Trainable( Variable ):
	'''
		Output of Trainable can be set freely, but should be set by optimizer
	'''
	def __init__( self, shape, initialValue: np.ndarray, **kwargs ):
		super( Trainable, self ).__init__( shape, **kwargs )

		self._value = initialValue

		assert initialValue.shape == shape

		_default_graph.trainable.append( self )

	def setValue( self, output:np.ndarray ):
		assert compareShape(output.shape, self._shape)
		self._value = output