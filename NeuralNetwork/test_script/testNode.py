import os
print(os.environ['PYTHONPATH'].split(os.pathsep))

from NeuralNetwork.CoreObject.Variable import PlaceHolder, Constant, Trainable
from NeuralNetwork.Operation.Basic import Multiply, Add, Substract, MatMul
from NeuralNetwork.CoreObject.Session import Session

import numpy as np

x = PlaceHolder( (2,2), name = 'x' )
y = PlaceHolder( (2,1), name = 'y' )
z = PlaceHolder( (2,1), name = 'z' )

xMulY = MatMul( x, y, name = 'mul' )
addZ = Add( xMulY, z, name = 'add' )

sess = Session()

xVal = np.array( [[1,0],[0,1]] )
yVal = np.array( [[1],[2]] )
zVal = np.array( [[-5], [5]] )

result = sess.run( [addZ], feedDict={x.Id:xVal, y.Id:yVal, z.Id:zVal} )

print( result )
print( xMulY, xMulY.getOutput() )
print( addZ.name, addZ.getOutput() )