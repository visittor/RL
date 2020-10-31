from NeuralNetwork.CoreObject.GraphComponent import GraphComponent
from NeuralNetwork.CoreObject.Node import Node
from NeuralNetwork.CoreObject.Variable import Variable, Constant, PlaceHolder
from NeuralNetwork.CoreObject.Session import Session
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

class _YGrad( Node ):

	def __init__( self, x:GraphComponent, val:float,  **kwargs ):
		super( _YGrad, self ).__init__( inputs=[x], **kwargs )
		self._val = val

	def computeShape( self, shape )->Tuple[int]:
		return shape

	def forward( self, x:np.ndarray )->np.ndarray:
		return np.ones( x.shape ) * self._val

	#
	#	Hack: since we don't use it anyway, I will not actually implement this function.
	#
	def backward( self, dL:np.ndarray )->np.ndarray:
		raise NotImplementedError

def gradients( yList, xList, grad_y:float = 1.0 ):
	
	allPath = {}
	for y in yList:
		assert not isinstance( y, Variable ), "Component yList cannot be %Variable%"
		path = _depthFirstSearch( y, xList )

		for k in path.keys():

			allPath.setdefault( k, [] ).extend( path[k] )

	gradDict = {}
	doBackwardList = []

	def construct( x ):
		if x in doBackwardList:
			return

		if x in yList:
			dL = _YGrad( x, grad_y, name="diff_{}".format(x.name) )
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

		doBackwardList.append( x )
		for i, input in enumerate( x.getInput() ):
			gradDict.setdefault( input, [] ).append(grads[i])

	for x in xList:
		construct( x )

	return [ Sum( gradDict[x], name="grad_{}".format(x.name) ) for x in xList ]

def findGradientNumerical( node:Node, inputNode:PlaceHolder, testData:List[np.ndarray], sess:Session, epsilon:float ):

	gradsVal = [ np.empty( data.shape ) for data in testData ]
	tmpData = [ data.copy() for data in testData ]

	for i, data in enumerate(testData):

		for idx in range(data.size):
			index = np.unravel_index( idx, data.shape )
			
			tmpData[i][index] -= epsilon
			val1 = sess.run( [node], feedDict={ inp.Id:data for inp,data in zip(inputNode, tmpData) } )[0]
			
			tmpData[i][index] += 2*epsilon
			val2 = sess.run( [node], feedDict={ inp.Id:data for inp,data in zip(inputNode, tmpData) } )[0]

			diff = np.sum(val2 - val1) / (2*epsilon)

			gradsVal[i][index] = diff

	return gradsVal

def check_gradient( node:Node, inputNode:PlaceHolder, testData:List[np.ndarray] ):

	epsilon = 1e-8

	grads = gradients( [node], inputNode )
	
	sess = Session()
	gradsVal_backprop = sess.run( grads, feedDict={ inp.Id:data for inp,data in zip(inputNode, testData) } )

	gradsVal_numeric = findGradientNumerical( node, inputNode, testData, sess, epsilon )
	print( gradsVal_numeric )
	print( gradsVal_backprop )
	print( '----')
	return [ backProp - numeric for backProp, numeric in zip(gradsVal_backprop, gradsVal_numeric) ]