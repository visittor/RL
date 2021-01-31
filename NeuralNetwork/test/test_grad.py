from RL import Tensor, fromNumpy, backward
from RL import BasicFunction as BF

import numpy as np

def printGraph( fn ):

	if fn is None:
		return

	if fn.nextFuncs is None :
		return

	for f in fn.nextFuncs:
		if f is not None:
			print( f[0], "({})".format(f[0].nIn), end=" " )
		else:
			print( f, end=" " )

	print()

	for f in fn.nextFuncs:
		if f is None:
			continue

		printGraph( f[0] )

w1 = fromNumpy( np.random.randn( 4,2 ) )

w2 = fromNumpy( np.random.randn( 4,4 ) )

w3 = fromNumpy( np.random.randn( 2,4 ) )

inp = fromNumpy( np.array( [0, 1] ).reshape(2,1) )

x = BF.max( BF.mm( w1, inp ), 0 )
x = BF.max( BF.mm( w2, x ), 0 )
x = BF.max( BF.mm( w3, x ), 0 )

# print( x.backward_fn )
# printGraph( x.backward_fn )

backward( x, np.ones( x.shape ) )

print( "Gradient w1:", w1.grad )
print( "Gradient w2:", w2.grad )
print( "Gradient w3:", w3.grad )
print( "Gradient inp:", inp.grad )