import os
print(os.environ['PYTHONPATH'].split(os.pathsep))

from NeuralNetwork.CoreObject.Variable import PlaceHolder, Constant, Trainable
from NeuralNetwork.Operation.Basic import Multiply, Add, Substract
from NeuralNetwork.CoreObject.Session import Session

import numpy as np

x = PlaceHolder( name = 'x' )
y = PlaceHolder( name = 'y' )
z = PlaceHolder( name = 'z' )

xMulY = Multiply( x, y, name = 'mul' )
addZ = Add( xMulY, z, name = 'add' )

subX = Substract( addZ, x, name = 'sub' )

sess = Session()

xVal = np.array( [5] )
yVal = np.array( [3] )
zVal = np.array( [-5] )

result = sess.run( subX, feedDict={x.Id:xVal, y.Id:yVal, z.Id:zVal} )

print( result )
print( xMulY, xMulY.getOutput() )
print( addZ.name, addZ.getOutput() )
print( subX, subX.getOutput() )