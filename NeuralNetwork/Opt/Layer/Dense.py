from NeuralNetwork.CoreObject.GraphComponent import GraphComponent
from NeuralNetwork.CoreObject.Variable import Constant, PlaceHolder, Trainable
from NeuralNetwork.CoreObject.Node import Node

from NeuralNetwork.Opt.Operation.Basic import Add, MatMul
from NeuralNetwork.Opt.Operation.Activation import Activation
from NeuralNetwork.Opt.Layer.AbstractLayer import Layer

import numpy as np

import math

from typing import Tuple

class Dense( Layer ):

	def __init__( self, units:int, activation:Activation, input_unit:int, initializer:str = "he", name = 'Dense' ):

		weight = np.random.randn( units, input_unit ) * math.sqrt( 2 / input_unit )

		self._w = Trainable( (units, input_unit), weight, name = '{}_'.format('weight') )
		self._b = Trainable( (units,1), np.zeros( (units,1) ), name = '{}_bias'.format('bias') )

		self._activation = activation

		self._name = name

	def _call( self, x:GraphComponent ):

		mul = MatMul( self._w, x )

		if self._activation is not None:
			out = Add( mul, self._b )
			out = self._activation( out, name = self._name )
			
		else:
			out = Add( mul, self._b, name = self._name )

		return out