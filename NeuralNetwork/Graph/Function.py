from .Tensor import Tensor
from collections.abc import Iterable

class TensorCarrier:

	def save( self, *tensors ):

		self._tensors = tensors
	
	def load( self ):

		if hasattr( self, "_tensors" ):
			return self._tensors
		
		return

class FunctionBase:

	def __init__( self ):

		self.nIn = 0
		self.nOut = 0

		self.nextFuncs = None

	@classmethod
	def apply( cls, *tensors ):

		assert hasattr( cls, "backward_cls" )

		backward_fn = cls.backward_cls()

		nextFuncs = []

		for tensor in tensors:

			if not isinstance( tensor, Tensor ):
				nextFuncs.append( None )
				continue

			nextFuncs.append( (tensor.backward_fn, tensor.index) )

		backward_fn.nextFuncs = tuple( nextFuncs )

		outputs = cls._calculate( backward_fn, *tensors )

		if not isinstance( outputs, (list, tuple, set) ):
			outputs = [ outputs ]

		backward_fn.nIn = len( tensors )
		backward_fn.nOut = len( outputs )

		finalOutputs = []

		for i, output in enumerate(outputs):

			if not isinstance( output, Tensor ):
				output = output.view( Tensor )

			output.backward_fn = backward_fn
			output.is_leaf = False

			output.index = i

			finalOutputs.append( output )

		if len( finalOutputs ) == 1:
			return finalOutputs[0]

		return finalOutputs

	def __str__( self ):
		return self.__class__.__name__

class BackwardFunction( TensorCarrier, FunctionBase ):

	def apply( self, *dL ):
		
		grads = self.__class__._calculate( self, *dL )

		if grads is None:
			return

		if not isinstance( grads, (list, tuple, set) ):
			grads = [ grads ]

		finalGrads = []

		for grad in grads:

			if not isinstance( grad, Tensor ):

				grad = grad.view( Tensor )

			finalGrads.append( grad )

		return finalGrads

class FunctionType( type ):

	def __init__( cls, name, bases, attrs ):

		backward_cls = type( name+'Backward', (BackwardFunction,), {} )

		backward_cls.forward_cls = cls
		cls.backward_cls = backward_cls

		cls._calculate = lambda ctx, *args : cls.forward( ctx, *args )
		backward_cls._calculate = lambda ctx, *args : cls.backward( ctx, *args )

class Function( TensorCarrier, FunctionBase, metaclass = FunctionType ):

	@staticmethod
	def forward( ctx, *inputs ):

		raise NotImplementedError

	@staticmethod
	def backward( ctx, *dL ):

		raise NotImplementedError