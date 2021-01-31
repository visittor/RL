from typing import List

import numpy as np

from .Function import Function

EPSILON = 1e-10

def _findGrad_numeric_singleElement( 
									func:Function, 
									index:int,
									coor:tuple,
									inputs : List
									):

		if not isinstance( inputs[index], np.ndarray ):
			inputs[ index ] += EPSILON

		else:
			inputs[ index ][ coor ] += EPSILON

		ctx = func.backward_cls()

		outs1 = func.forward( ctx, *inputs )

		if not isinstance( inputs[index], np.ndarray ):
			inputs[ index ] -= 2 * EPSILON

		else:
			inputs[ index ][ coor ] -= 2 * EPSILON

		outs2 = func.forward( ctx, *inputs )

		totalDiff = 0

		for out1, out2 in zip( outs1, outs2 ):

			diff = (out1 - out2) / (2*EPSILON)

			totalDiff += diff.sum()

		return totalDiff

def findGrad_numeric( func : Function, index : int, *inputs ):

	dL = np.zeros_like( inputs[index] )

	nElement = dL.size

	for i in range( nElement ):
		
		coor = np.unravel_index( i, dL.shape )

		totalDiff = _findGrad_numeric_singleElement( 
													func, 
													index, 
													coor, 
													list(inputs)
													)

		dL[ coor ] = totalDiff

	return dL

def checkGrad( func : Function, *inputs ):

	ctx = func.backward_cls()

	outs = func.forward( ctx, *inputs )

	if not isinstance( outs, (tuple, list) ):
		outs = [ outs ]

	dLs = [ np.ones_like( out ) for out in outs ]

	grads = func.backward( ctx, *dLs )

	if not isinstance( grads, (tuple, list) ):
		grads = [ grads ]

	for i, inp in enumerate(inputs):
		if not isinstance( inp, np.ndarray ):
			inp = np.array( inp )

		dL = findGrad_numeric( func, i, *inputs )

		print( f"TEST GRAD INDEX {i}" )
		print ( f"DL autograd : {grads[i]}")
		print ( f"DL numerical : {dL}")
		print( f"DIFFERENT IS \n{grads[i] - dL}" )