from .GraphComponent import GraphComponent
from .Graph import _default_graph

from typing import List, Dict

import numpy as np

class Node( GraphComponent ):

	def __init__( self, inputs:List[GraphComponent] = None, **kwargs ):
		super( Node, self ).__init__( **kwargs )

		self._input: List[GraphComponent] = inputs if inputs is not None else []

		_default_graph.node.append( self )

	def getInput( self )->List[GraphComponent]:
		return self._input

	def forward( self )->np.ndarray:
		raise NotImplementedError

	def backward( self )->np.ndarray:
		raise NotImplementedError