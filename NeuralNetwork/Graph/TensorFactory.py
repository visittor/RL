from .Tensor import Tensor
from .Function import Function

import numpy as np

class AccumulatedGrad( Function ):

	@staticmethod
	def forward( ctx, *inputs ):

		assert False, "No one should call this funcking function."

	@staticmethod
	def backward( ctx, dL ):

		if not ctx.var.compute_grad:
			return

		if ctx.var.grad is None:
			ctx.var.grad = dL.copy().view( Tensor )

		else:
			ctx.var.grad += dL

		return ctx.var.grad

def fromNumpy( arr ):

	if not isinstance( arr, np.ndarray ):
		arr = np.array( arr )

	tensor = arr.view( Tensor )

	tensor.backward_fn = AccumulatedGrad.backward_cls()
	tensor.backward_fn.var = tensor

	tensor.is_leaf = True
	tensor.index = 0
	tensor.grad = None

	return tensor