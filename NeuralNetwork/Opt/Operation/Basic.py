from NeuralNetwork.CoreObject.Node import Node
from NeuralNetwork.CoreObject.GraphComponent import GraphComponent
from NeuralNetwork.CoreObject.Variable import Constant

import numpy as np

from typing import List, Tuple
from itertools import zip_longest
from functools import reduce

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

	class __SumBroadCast( Node ):

		def __init__( self, x, axis ):
			self._axis = axis
			super().__init__( inputs=[x] )
		
		def computeShape( self, shape ):
			return tuple( shape[i] if i != self._axis else 1 for i in range(len(shape)) )

		def forward( self, x:np.ndarray )->np.ndarray:
			return np.sum( x, axis = self._axis )

		#
		#	HACK: since we don't use it anyway, I will not actually implement this function.
		#
		def backward( self, dL:np.ndarray )->np.ndarray:
			raise NotImplementedError

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
		if dX.shape != self._input[0].shape:
			dX = self.__SumBroadCast( dX, 0 )

		dY = MatMul( Transpose(self._input[0]), dL, name="diff_{}".format(self._input[1].name) )
		print( "dY", dY.shape )
		if dY.shape != self._input[1].shape:
			print( "Do sum broad")
			dY = self.__SumBroadCast( dY, 0 )

		return [dX, dY]

class ElementwiseMixin:

	class __SumBroadCast( Node ):

		def __init__( self, x, axis ):
			self._axis = axis
			super().__init__( inputs=[x] )
		
		def computeShape( self, shape ):
			return tuple( shape[i] if i != self._axis else 1 for i in range(len(shape)) )

		def forward( self, x:np.ndarray )->np.ndarray:
			return np.sum( x, axis = self._axis )

		#
		#	HACK: since we don't use it anyway, I will not actually implement this function.
		#
		def backward( self, dL:np.ndarray )->np.ndarray:
			raise NotImplementedError

	def computeShape( self, shape1, shape2 )->Tuple[int]:
		assert all( m==n or m==1 or n==1 for m,n in zip(shape1[::-1],shape2[::-1]) ), \
				"Invalid shape"
		
		shape = tuple( 	m if m!=1 and n!= 1 else int(m*n) \
						for m, n in zip_longest( shape1[::-1], shape2[::-1], fillvalue=1 ) )[::-1]

		return shape

	def doSumBroadcast( self, node:GraphComponent, inputShape, outputShape ):
		doSumBroadCastAxe = None
		for i, (sx, sl) in enumerate(zip_longest(inputShape[::-1], outputShape[::-1], fillvalue=1)):
			if sl != sx and sx == 1:
				doSumBroadCastAxe = max(len(inputShape), len(outputShape)) - i - 1
				break
		if doSumBroadCastAxe is not None:
			dx = self.__SumBroadCast( node, doSumBroadCastAxe )
			return dx

		return node

class Multiply( ElementwiseMixin, Node ):

	def __init__( self, x:GraphComponent, y:GraphComponent,
				**kwargs ):
		super( Multiply, self ).__init__( inputs=[x, y], **kwargs )

	def forward( self, x:np.ndarray, y:np.ndarray )->np.ndarray:
		return np.multiply( x, y )

	def backward( self, dL:GraphComponent )->List[GraphComponent]:

		dX = Multiply( dL, self._input[1], name="diff_{}".format(self._input[0].name) )
		dY = Multiply( self._input[0], dL, name="diff_{}".format(self._input[1].name) )

		return [dX, dY]

class Power( Node ):

	def __init__( self, x:GraphComponent, a:float, **kwargs ):
		super().__init__( inputs=[x], **kwargs )
		self._a = a
	
	def computeShape( self, shape1 )->Tuple[int]:
		return shape1

	def forward( self, x:np.ndarray )->np.ndarray:
		return np.power( x, self._a )

	def backward( self, dL:GraphComponent )->List[GraphComponent]:

		d = Multiply( Constant((1,), np.array([self._a])),
						Power(self._input[0], self._a - 1) )
		return [ Multiply( dL, d, name="diff_{}".format(self._input[0].name) ) ]

class Log( Node ):

	def __init__( self, x:GraphComponent, **kwargs ):
		super().__init__( inputs=[x], **kwargs )
	
	def computeShape( self, shape1 )->Tuple[int]:
		return shape1

	def forward( self, x:np.ndarray )->np.ndarray:
		return np.log( x )

	def backward( self, dL:GraphComponent )->List[GraphComponent]:
		d = Divide( Constant((1,), np.array([1]) ), self._input[0] )
		return [ Multiply( dL, d, name="diff_{}".format(self._input[0].name) ) ]

class Divide( ElementwiseMixin, Node ):

	def __init__( self, x:GraphComponent, y:GraphComponent, **kwargs ):
		super().__init__( inputs=[x, y], **kwargs )

	def forward( self, x:np.ndarray, y:np.ndarray )->np.ndarray:
		return x / y

	def backward( self, dL:GraphComponent )->List[GraphComponent]:

		dx = Divide( Constant((1,), np.ones((1,))), self._input[1] )
		dx = Multiply( dL, dx, name="diff_{}".format(self._input[0].name) )

		dy = Multiply( Constant((1,), np.array([-1])),
						Divide( self._input[0], Power(self._input[1], 2.0) ) )
		dy = Multiply( dL, dy, name="diff_{}".format(self._input[1].name) )

		return [dx, dy]

class Add( ElementwiseMixin, Node ):

	def __init__( self, x:GraphComponent, y:GraphComponent,
				**kwargs ):
		super( Add, self ).__init__( inputs=[x, y], **kwargs )

	def forward( self, x:np.ndarray, y:np.ndarray )->np.ndarray:
		return x + y

	def backward( self, dL:GraphComponent )->List[GraphComponent]:
		dx = Buffer( dL, name="diff_{}".format(self._input[0].name) )
		dx = self.doSumBroadcast( dx, self._input[0].shape, dL.shape )

		dy = Buffer( dL, name="diff_{}".format(self._input[1].name) )
		dy = self.doSumBroadcast( dy, self._input[1].shape, dL.shape )

		return [dx, dy]

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

class Substract( ElementwiseMixin, Node ):

	def __init__( self, x:GraphComponent, y:GraphComponent,
				**kwargs ):
		super( Substract, self ).__init__( inputs=[x, y], **kwargs )

	def forward( self, x:np.ndarray, y:np.ndarray )->np.ndarray:
		return x - y

	def backward( self, dL:GraphComponent )->List[GraphComponent]:

		d = Buffer( dL, name="diff_{}".format(self._input[0].name) )
		dY = Multiply( dL, Constant( (1,), np.array([-1]) ), name="diff_{}".format(self._input[1].name) )

		return [d, dY]

class Max( ElementwiseMixin, Node ):

#	HACK : For create node that do derivetive of max
	class __DiffMax( ElementwiseMixin, Node ):
		def __init__( self, x:GraphComponent, y:GraphComponent, **kwargs ):
			super().__init__( inputs=[x,y], **kwargs )

		def forward( self, x:np.ndarray, y:np.ndarray )->np.ndarray:
			return x >= y

		#
		#	HACK: since we don't use it anyway, I will not actually implement this function.
		#
		def backward( self, dL:np.ndarray )->np.ndarray:
			raise NotImplementedError

	def __init__( self, x:GraphComponent, y:GraphComponent,
				**kwargs ):
		super( Max, self ).__init__( inputs=[x, y], **kwargs )

	def forward( self, x:np.ndarray, y:np.ndarray )->np.ndarray:
		return np.maximum( x, y )

	def backward( self, dL:GraphComponent )->List[GraphComponent]:

		dX = Multiply( dL, self.__DiffMax( self._input[0], self._input[1] ), name="diff_{}".format(self._input[0].name) )
		dY = Multiply( dL, self.__DiffMax( self._input[1], self._input[0] ), name="diff_{}".format(self._input[1].name) )

		return [dX, dY]

class Reshape( Node ):

	def __init__( self, x:GraphComponent, shape:Tuple[int], **kwargs ):
		self._shape = shape
		super( Reshape, self ).__init__( inputs=[x], **kwargs )

	def computeShape( self, shape1 )->Tuple[int]:
		totalElement1 = reduce((lambda x, y: x * y), shape1)
		totalElement2 = reduce((lambda x, y: x * y), self._shape)
		assert totalElement1 == totalElement2 or totalElement1<0 or totalElement2 < 0 

		return self._shape
	
	def forward( self, x:np.ndarray )->np.ndarray:
		return x.reshape( self._shape )
	
	def backward( self, dL:GraphComponent )->GraphComponent:
		dx = Reshape( dL, self._input[0].shape )

		return [dx]