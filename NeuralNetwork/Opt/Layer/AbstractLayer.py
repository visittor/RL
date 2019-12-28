from NeuralNetwork.CoreObject.GraphComponent import GraphComponent

from typing import List


class Layer( object ):

	def __call__( self, *args ):

		return self._call( *args )

	def _call( self, inputs:List[GraphComponent] )->GraphComponent:
		raise NotImplementedError