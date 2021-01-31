import math

import numpy as np

from .Module import Module
from ..BasicFunction import BasicFunction as BF
from ..Graph import fromNumpy

class Dense( Module ):

	def __init__( self, nIn : int, nOut : int, bias: bool = False ):

		super( Dense, self ).__init__()

		stdev = math.sqrt( 2. / nIn )
		weight = np.random.randn( nOut, nIn ) * stdev
		self.weight_par = fromNumpy( weight )

		bias = np.zeros( ( nOut, 1 ) )
		self.bias_par = fromNumpy( bias )

	def forward( self, x ):

		return BF.add( BF.mm( self.weight_par, x ), self.bias_par )