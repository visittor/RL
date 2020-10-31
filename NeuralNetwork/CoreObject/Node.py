from .GraphComponent import GraphComponent
from .Graph import _default_graph

from typing import List, Dict, Tuple

import numpy as np

from abc import ABCMeta, abstractmethod

class Node( GraphComponent, metaclass = ABCMeta ):
	'''
		This type of GraphComponent need inputs. 
		
		func:
			forward: MUST BE IMPLEMENTED. it takes
					numpy array(s) which output of this Node
					inputs. it will return calculated numpy array
					which will be this node output later.
			
			backward: MUST BE IMPLEMENTED. Return GraphComponents
						that will calculate gradient of its input nodes
			
			computeShape: MUST BE IMPLEMENTED. compute dimension of output
							of this node base on input nodes.
				
	'''
	def __init__( self, inputs:List[GraphComponent] = None, **kwargs ):
		super( Node, self ).__init__( **kwargs )

		self._input: List[GraphComponent] = inputs if inputs is not None else []

		self._shape = self.computeShape( *[x.shape for x in self._input] )

		_default_graph.node.append( self )

	@abstractmethod
	def computeShape( self, *shape )->Tuple[int]:
		raise NotImplementedError

	def getInput( self )->List[GraphComponent]:
		return self._input

	@abstractmethod
	def forward( self )->np.ndarray:
		raise NotImplementedError

	@abstractmethod
	def backward( self, dL:GraphComponent )->List[GraphComponent]:
		raise NotImplementedError

	@property
	def shape( self ):
		return self._shape