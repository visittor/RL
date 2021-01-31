from typing import Iterable, Callable, Any

import numpy as np

from ..nn import Module
from ..Graph import Tensor
from ..Graph import backward

class GradientDescend:

	def __init__( self,
				parameters : Iterable[Tensor],
				lr : float = 0.01
				):

		self.parameters = parameters

		self.lr = lr

	def update( self, loss_fn : Callable[[], Tensor] ):

		loss = loss_fn()

		backward( loss, np.ones( loss.shape ) )

		for param in self.parameters:

			if not param.compute_grad:
				continue

			param -= self.lr * param.grad

	def clearGrad( self ):

		for param in self.parameters:

			param.grad = None