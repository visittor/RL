import numpy as np

from RL import nn
from RL import Tensor, fromNumpy, backward
from RL import GradientDescend

class Model( nn.Module ):

	def __init__( self ):

		super( Model, self ).__init__()

		self.l1_module = nn.Dense( 2, 4 )
		self.l2_module = nn.Dense( 4, 4 )
		self.l3_module = nn.Dense( 4, 2 )
		self.l4_module = nn.Dense( 2, 1 )

	def forward( self, x : Tensor ):

		x = self.l1_module.forward( x )
		x = nn.relu( x )
		x = self.l2_module.forward( x )
		x = nn.relu( x )
		x = self.l3_module.forward( x )
		x = nn.relu( x )
		x = self.l4_module.forward( x )
		x = nn.sigmoid( x )

		return x

inpVal = np.random.randn( 500, 2, 1 )

outVal = np.zeros( ( 500, 1, 1 ) )
outVal[ inpVal[:, 0, 0] * inpVal[:, 1, 0] < 0 ] = 1

print( outVal.sum() )
input()

inpVal = fromNumpy( inpVal )
outVal = fromNumpy( outVal )

inpVal.compute_grad = False
outVal.compute_grad = False

def loss( model, x, y ):

	x = model.forward( x )

	l = nn.logloss( x, y )
	print( l )
	return l

model = Model()

opt = GradientDescend( model.getAllParameters(), lr = 0.1 )

for i in range( 500 ):

	l_fn = lambda : loss( model, inpVal, outVal )

	opt.update( l_fn )

	opt.clearGrad()

	# backward( l, np.ones( l.shape ) )

x = model.forward( inpVal )

for i in range( 100 ):
	print( inpVal[i].flatten(), x[i], outVal[i].flatten() )
