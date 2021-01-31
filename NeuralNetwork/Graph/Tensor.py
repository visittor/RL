import numpy as np

class Tensor( np.ndarray ):

	def __new__(subtype, shape, dtype=float, buffer=None, offset=0,
				strides=None, order=None, info=None):
		# Create the ndarray instance of our type, given the usual
		# ndarray input arguments.  This will call the standard
		# ndarray constructor, but return an object of our type.
		# It also triggers a call to InfoArray.__array_finalize__
		obj = super(Tensor, subtype).__new__(subtype, shape, dtype,
												buffer, offset, strides,
												order)

		obj._init()

		# Finally, we must return the newly created object:
		return obj

	def __array_finalize__( self, obj ):

		if obj is None:
			return

		self._init()

	def _init( self ):

		self.backward_fn = None
		self.is_leaf = True

		self.compute_grad = True

		self.index = 0

		self.grad = None