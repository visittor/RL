from .Module import Module
from ..BasicFunction import BasicFunction as BF

class ReLU( Module ):

	def forward( self, x ):

		return BF.max( x, 0 )

class LeakyReLU( Module ):

	def forward( self, x ):

		return BF.max( x, BF.multiply( 0.1, x ) )

class Sigmoid( Module ):

	def forward( self, x ):

		return BF.divide( 1, BF.add( 1, BF.exp( BF.multiply( x, -1 ) ) ) )

class Tanh( Module ):

	def forward( self, x ):

		ex = BF.exp( x )
		eMinusX = BF.exp( BF.multiply( -1, x ) )

		return BF.divide( BF.substract( ex, eMinusX ), BF.add( eMinusX, ex ) )

relu = ReLU().forward
leaky_relu = LeakyReLU().forward
sigmoid = Sigmoid().forward
tanh = Tanh().forward