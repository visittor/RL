from .Tensor import Tensor
from .Function import Function

from typing import List, Tuple, Dict

import numpy as np

def backward( tensor : Tensor, dL : np.ndarray ):

	def _backward( func : Function, index : int , dL, waitQueue : Dict ):

		nextFuncs = func.nextFuncs

		if not isinstance( dL, ( list, tuple ) ):
			dL = [ dL ]

		# print( "Apply:", func, dL )
		dL = func.apply( *dL )

		if nextFuncs is None:
			return

		# print( "nextFuncs:", nextFuncs )

		for nextItem, d in zip( nextFuncs, dL ):

			if nextItem is None:
				continue

			nextF, f_index = nextItem

			if nextF.nOut > 1:

				# TODO:
				if nextF not in waitQueue:
					waitQueue[ nextF ] = {}

				waitQueue[ nextF ][ index ] = d

				if len( waitQueue[ nextF ] ) == nextF.nOut:
					
					_backwardMultiOutput( nextF, waitQueue )

				continue

			_backward( nextF, f_index, d, waitQueue )

	def _backwardMultiOutput( func : Function, waitQueue : Dict ):

		dLDict = waitQueue[ func ]

		dL = []

		for i in range( func.nOut ):

			dL.append( dLDict[ i ] )

		dL = func.apply( *dL )

		if not isinstance( dL, ( list, tuple ) ):
			dL = [ dL ]

		for (nextF, f_index), d in zip( func.nextFuncs, dL ):
			
			_backward( nextF, f_index, d, waitQueue )

	waitQueue = {}

	if not isinstance( dL, ( list, tuple ) ):
		dL = [ dL ]

	_backward( tensor.backward_fn, tensor.index, dL, waitQueue )