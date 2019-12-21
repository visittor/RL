from ..CoreObject.GraphComponent import GraphComponent
from ..CoreObject.Node import Node
from ..CoreObject.Variable import Variable, Constant
from .Basic import Sum

import numpy as np
from typing import Dict, Tuple, List

def _depthFirstSearch( start, goals )->Dict:

	path = {}
	foundList = set()

	def addPath( n:GraphComponent ):

		if isinstance( n, Node ):
			for input in n.getInput():
				path.setdefault( input, [] ).append( n )
				addPath( input )

		if n in goals:
			foundList.add( n )

	addPath( start )

	if len(foundList) == 0:
		return {}

	return path


def gradients( yList, xList ):
	
	allPath = {}
	for y in yList:
		assert not isinstance( y, Variable ), "Component yList cannot be %Variable%"
		path = _depthFirstSearch( y, xList )

		for k in path.keys():

			allPath.setdefault( k, [] ).extend( path[k] )

	gradDict = {}

	def construct( x ):

		if x in gradDict:
			return

		if x in yList:
			dL = _YGrad( x, name="diff_{}".format(x.name) )
			grads = x.backward( dL )
			gradDict.setdefault(x, []).append( dL )

		else:

			assert x in allPath, "{} is not connected".format( x.name )

			for connected in allPath[ x ]:
				
				construct( connected )

			if isinstance( x, Variable ):
				return

			dL = Sum( gradDict[x] )

			grads = x.backward( dL )

		for i, input in enumerate( x.getInput() ):
			gradDict.setdefault( input, [] ).append(grads[i])

	for x in xList:
		construct( x )

	return [ Sum( gradDict[x], name="grad_{}".format(x.name) ) for x in xList ]

class _YGrad( Node ):

	def __init__( self, x:GraphComponent, **kwargs ):
		super( _YGrad, self ).__init__( inputs=[x], **kwargs )

	def computeShape( self, shape )->Tuple[int]:
		return shape

	def forward( self, x:np.ndarray )->np.ndarray:
		return np.ones( x.shape )