from ..BasicFunction import BasicFunction as BF

class LogLoss:

	@staticmethod
	def forward( x, y ):

		p = BF.multiply( y, BF.log( x ) )
		n = BF.multiply( BF.substract( 1, y ), BF.log( BF.substract( 1, x ) ) )

		l = BF.add( p, n )
		# print(l.shape, p.shape, n.shape, x.shape, y.shape)
		n = l.shape[0]
	
		l = BF.reduce_sum( l, 0 )

		l = BF.divide( l, -1 * n )

		return l

logloss = LogLoss.forward

