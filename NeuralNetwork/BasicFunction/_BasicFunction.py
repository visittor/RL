from ..Graph.Function import Function
from ..Graph.Tensor import Tensor
import numpy as np

def _reduceMatchNDim( a, target ):
	if isinstance( target, np.ndarray ):

		assert a.ndim >= target.ndim

		while a.ndim > target.ndim:
			a = np.sum( a, axis = 0 )

	else:
		a = np.sum( a )

	return a

class MatMul( Function ):

	@staticmethod
	def forward( ctx, a, b ):

		c = np.matmul( a, b )
		c = c.view( Tensor )

		ctx.save( a, b )
		# print( f"a : {a.shape}, b : {b.shape}, c : {c.shape}" )
		return c

	@staticmethod
	def backward( ctx, dL ):

		savedTensor = ctx.load()

		a = savedTensor[ 0 ]
		b = savedTensor[ 1 ]

		aT = a.T if a.ndim == 2 else a.transpose( 0, 2, 1 )
		bT = b.T if b.ndim == 2 else b.transpose( 0, 2, 1 )

		# print( f"dL : {dL.shape}, aT : {aT.shape}, bT : {bT.shape}" )

		dX = np.matmul( dL, bT )
		dY = np.matmul( aT, dL )

		dX = _reduceMatchNDim( dX, a )
		dY = _reduceMatchNDim( dY, b )

		return ( dX, dY )

class Multiply( Function ):

	@staticmethod
	def forward( ctx, a, b ):

		ctx.save( a, b )

		return np.multiply( a, b )

	@staticmethod
	def backward( ctx, dL ):

		a, b = ctx.load()

		dA = np.multiply( dL, b )
		dB = np.multiply( a, dL )

		dA = _reduceMatchNDim( dA, a )
		dB = _reduceMatchNDim( dB, b )

		return dA, dB

class Divide( Function ):

	@staticmethod
	def forward( ctx, a, b ):

		ctx.save( a, b )

		return np.divide( a, b )

	@staticmethod
	def backward( ctx, dL ):

		a, b = ctx.load()

		dA = np.divide( dL, b )
		dB = np.divide( -a, np.power( b, 2 ) )
		dB = np.multiply( dB, dL )

		dA = _reduceMatchNDim( dA, a )
		dB = _reduceMatchNDim( dB, b )

		return dA, dB

class Add( Function ):

	@staticmethod
	def forward( ctx, a, b ):

		c = a + b
		c.view( Tensor )

		ctx.save( a, b )

		return c

	@staticmethod
	def backward( ctx, dL ):

		dL1 = dL.copy().view( Tensor )

		dL2 = dL.copy().view( Tensor )

		a, b = ctx.load()

		dL1 = _reduceMatchNDim( dL1, a )
		dL2 = _reduceMatchNDim( dL2, b )

		return ( dL1, dL2 )

class Substract( Function ):

	@staticmethod
	def forward( ctx, a, b ):

		ctx.save( a, b )

		return np.subtract( a, b )

	@staticmethod
	def backward( ctx, dL ):

		dL1 = dL.copy().view( Tensor )

		dL2 = -1 * dL.copy().view( Tensor )

		a, b = ctx.load()

		dL1 = _reduceMatchNDim( dL1, a )
		dL2 = _reduceMatchNDim( dL2, b )

		return ( dL1, dL2 )

class Exp( Function ):

	@staticmethod
	def forward( ctx, a ):

		out = np.exp( a )

		ctx.save( a, out )

		return out
	
	@staticmethod
	def backward( ctx, dL ):

		a, out = ctx.load()

		dL = np.multiply( out, dL )

		return dL

class Log( Function ):

	@staticmethod
	def forward( ctx, a ):

		ctx.save( a )

		return np.log( a )

	@staticmethod
	def backward( ctx, dL ):

		a = ctx.load()[0]

		return np.divide( dL, a )

class Max( Function ):

	@staticmethod
	def forward( ctx, a, b ):

		mask = (a >= b)
		# print( f"a: {a}\nb: {b}" )
		out = np.maximum( a, b )

		ctx.save( mask, a, b )

		return out

	@staticmethod
	def backward( ctx, dL ):

		mask, a, b = ctx.load()
		invMask = ~mask

		dL1 = dL.copy()
		dL1[invMask] = 0

		dL2 = dL.copy()
		dL2[mask] = 0

		dL1 = _reduceMatchNDim( dL1, a )
		dL2 = _reduceMatchNDim( dL2, b )

		return dL1, dL2

class ReduceSum( Function ):

	@staticmethod
	def forward( ctx, x, axis ):

		ctx.save( x, axis )
		# print( "ReduceSum for", x.shape )
		return np.sum( x, axis=axis )

	@staticmethod
	def backward( ctx, dL ):

		x, axis = ctx.load()

		dL = np.expand_dims( dL, axis=axis )
		dL = np.repeat( dL, x.shape[axis], axis = axis )

		# print( f"DL : {dL.shape}, x : {x.shape}" )

		return dL