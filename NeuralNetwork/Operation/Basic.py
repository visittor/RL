from ..CoreObject.Node import Node
from ..CoreObject.GraphComponent import GraphComponent
from ..CoreObject.Variable import Constant

import numpy as np

from typing import List, Tuple

class Buffer( Node ):

	def __init__( self, x:GraphComponent, **kwargs ):

		super( Buffer, self ).__init__( inputs=[x], **kwargs )

	def computeShape( self, shape:Tuple ):
		return shape

	def forward( self, x:np.ndarray )->np.ndarray:
		return x
	
	def backward( self, dL:GraphComponent )->List[GraphComponent]:
		return [dL]

class Transpose( Node ):

	def __init__( self, x:GraphComponent, **kwargs ):

		super( Transpose, self ).__init__( inputs=[x], **kwargs )

	def computeShape( self, shape:Tuple )->Tuple:
		assert 1< len(shape) <= 3, "shape for transpose is invalid"
		if len( shape ) == 2:
			return shape[::-1]

		return (shape[0], shape[2], shape[1])

	def forward( self, x:np.ndarray )->np.ndarray:
		if x.ndim == 2:
			return np.transpose( x )
		else:
			return np.transpose( x, axes=(0, 2, 1))

	def backward( self, dL:GraphComponent )->List[GraphComponent]:
		return [ Transpose( dL ) ]

class MatMul( Node ):
	
	def __init__( self, x:GraphComponent, y:GraphComponent,
				**kwargs ):
		super( MatMul, self ).__init__( inputs=[x, y], **kwargs )

	def computeShape( self, shape1:Tuple, shape2:Tuple )->Tuple:
		assert 1< len(shape1) <= 3, "shape for transpose is invalid"
		assert 1< len(shape2) <= 3, "shape for transpose is invalid"
		
		ndim = max( len(shape1), len(shape2) )
		
		if ndim == 2:
			return ( shape1[0], shape2[1] )

		if len(shape1) == len(shape2):
			assert 	shape1[0] == -1 or \
					shape2[0] == -1 or \
					shape1[0] == shape2[0], "Invalid shape"

			m = shape1[0] if shape1[0] == -1 else shape2[0]

		else:
			m = shape1[0] if len(shape1) == 3 else shape2[0]

		n = shape1[0] if len(shape1) == 2 else shape1[1]
		k = shape2[1] if len(shape2) == 2 else shape2[2]

		return (m, n, k)

	def forward( self, x:np.ndarray, y:np.ndarray )->np.ndarray:
		return np.matmul( x, y )

	def backward( self, dL:GraphComponent )->List[GraphComponent]:

		dX = MatMul( dL, Transpose(self._input[1]), name="diff_{}".format(self._input[0].name) )
		dY = MatMul( Transpose(self._input[0]), dL, name="diff_{}".format(self._input[1].name) )

		return [dX, dY]

class Multiply( Node ):

	def __init__( self, x:GraphComponent, y:GraphComponent,
				**kwargs ):
		super( Multiply, self ).__init__( inputs=[x, y], **kwargs )

	def computeShape( self, shape1, shape2 )->Tuple[int]:
		assert len(shape1) == len(shape2), "Invalid shape"
		assert shape1 == shape2, "Invalid shape"

		return shape1

	def forward( self, x:np.ndarray, y:np.ndarray )->np.ndarray:
		return np.multiply( x, y )

	def backward( self, dL:GraphComponent )->List[GraphComponent]:

		dX = Multiply( dL, self._input[1], name="diff_{}".format(self._input[0].name) )
		dY = Multiply( self._input[0], dL, name="diff_{}".format(self._input[1].name) )

		return [dX, dY]

class Add( Node ):

	def __init__( self, x:GraphComponent, y:GraphComponent,
				**kwargs ):
		super( Add, self ).__init__( inputs=[x, y], **kwargs )

	def computeShape( self, shape1, shape2 )->Tuple[int]:
		assert len(shape1) == len(shape2) or len(shape1)-1 == len(shape2), "Invalid shape"
		
		if len(shape1) == len(shape2):
			assert shape1 == shape2, "Invalid shape"
			return shape1
		
		assert shape1[1:] == shape2, "Invalid shape"
		return shape1

	def forward( self, x:np.ndarray, y:np.ndarray )->np.ndarray:
		return x + y

	def backward( self, dL:GraphComponent )->List[GraphComponent]:
		d = Buffer( dL, name="diff_{}".format(self._input[0].name) )
		return [d, d]

class Sum( Node ):

	def __init__( self, xs:List[GraphComponent], **kwargs ):

		super().__init__( inputs=xs, **kwargs )

	def computeShape( self, *shapes )->Tuple[int]:
		
		for shape in shapes:

			for s1, s2 in zip( shape, shapes[0] ):
				
				assert s1 == -1 or s2 == -1 or s1 == s2, "Ivalid shape"

		return shapes[0]

	def forward( self, *xs )->np.ndarray:
		return sum( xs )

	def backward( self, dL:GraphComponent )->List[GraphComponent]:
		d = Buffer( dL, name="diff_{}".format(self._input[0].name) )
		return [d for i in range( len( self._input ) )]

class Substract( Node ):

	def __init__( self, x:GraphComponent, y:GraphComponent,
				**kwargs ):
		super( Substract, self ).__init__( inputs=[x, y], **kwargs )

	def computeShape( self, shape1, shape2 )->Tuple[int]:
		assert len(shape1) == len(shape2) or len(shape1)-1 == len(shape2), "Invalid shape"
		
		if len(shape1) == len(shape2):
			assert shape1 == shape2, "Invalid shape"
			return shape1
		
		assert shape1[1:] == shape2, "Invalid shape"
		return shape1

	def forward( self, x:np.ndarray, y:np.ndarray )->np.ndarray:
		return x - y

	def backward( self, dL:GraphComponent )->List[GraphComponent]:

		d = Buffer( dL, name="diff_{}".format(self._input[0].name) )
		dY = Multiply( dL, Constant( (1,), np.array([-1]) ), name="diff_{}".format(self._input[1].name) )

		return [d, dY]

class Max( Node ):

#	HACK : For create node that do derivetive of max
	class __DiffMax( Node ):
		def __init__( self, x:GraphComponent, y:GraphComponent, **kwargs ):
			super().__init__( inputs=[x,y], **kwargs )

		def computeShape( self, shape:Tuple ):
			return shape

		def forward( self, x:np.ndarray, y:np.ndarray )->np.ndarray:
			return x >= y

	def __init__( self, x:GraphComponent, y:GraphComponent,
				**kwargs ):
		super( Max, self ).__init__( inputs=[x, y], **kwargs )

	def computeShape( self, shape:Tuple ):
		return shape

	def forward( self, x:np.ndarray, y:np.ndarray )->np.ndarray:
		return np.maximum( x, y )

	def backward( self, dL:GraphComponent )->List[GraphComponent]:

		dX = Multiply( dL, self.__DiffMax( self._input[0], self._input[1] ), name="diff_{}".format(self._input[0].name) )
		dY = Multiply( dL, self.__DiffMax( self._input[1], self._input[0] ), name="diff_{}".format(self._input[1].name) )

		return [dX, dY]

# class Exponent( Node ):

# 	def __init__( self, x:GraphComponent, y:GraphComponent,
# 				**kwargs ):
# 		super( Exponent, self ).__init__( inputs=[x, y], **kwargs )

# 	def forward( self, x:np.ndarray, y:np.ndarray )->np.ndarray:
# 		return np.power( x, y )

# 	def backward( self, dL:np.ndarray, prevX:np.ndarray, 
# 		prevY:np.ndarray )->List[np.ndarray]:

# 		dX = np.multiply( prevY, np.power( prevX, prevY - 1 ) )
# 		dX = np.multiply( dL, dX )

# 		dY = np.multiply(np.power( prevX, prevY ), np.log( prevX ) )
# 		dY = np.multiply( dL, dY )

# 		return [dX, dY]

class Sigmoid( Node ):

#	HACK : For create node that do derivetive of sigmoid
	class __DiffSigmoid( Node ):
		def __init__( self, x:GraphComponent, **kwargs ):
			super().__init__( inputs=[x], **kwargs )

		def computeShape( self, shape:Tuple ):
			return shape

		def forward( self, x:np.ndarray )->np.ndarray:
			
			sig = np.power( np.e, x )

			dX = np.multiply( sig, 1 - sig )

			return [dX]

	def __init__( self, x:GraphComponent, **kwargs ):
		super( Sigmoid, self ).__init__( inputs=[x], **kwargs )

	def computeShape( self, shape:Tuple ):
		return shape

	def forward( self, x:np.ndarray )->np.ndarray:
		return np.power( np.e, x )

	def backward( self, dL:GraphComponent )->List[GraphComponent]:
		
		dX = Multiply( dL, self.__DiffSigmoid( self._input[0] ), name="diff_{}".format(self._input[0].name) )

		return [dX]

class Tanh( Node ):

#	HACK : For create node that do derivetive of tanh
	class __DiffTanh( Node ):
		def __init__( self, x:GraphComponent, **kwargs ):
			super().__init__( inputs=[x], **kwargs )

		def computeShape( self, shape:Tuple ):
			return shape

		def forward( self, x:np.ndarray )->np.ndarray:
			
			tanh = np.tanh( x )

			dX = 1 - np.power( tanh, 2 )

			return [dX]

	def __init__( self, x:GraphComponent, **kwargs ):
		super( Tanh, self ).__init__( inputs=[x], **kwargs )

	def computeShape( self, shape:Tuple ):
		return shape

	def forward( self, x:np.ndarray )->np.ndarray:
		return np.tanh( x )

	def backward( self, dL:GraphComponent )->List[GraphComponent]:

		dX = Multiply( dL, self.__DiffTanh( self._input[0] ),name="diff_{}".format(self._input[0].name) )

		return [dX]