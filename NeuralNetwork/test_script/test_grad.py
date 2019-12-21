from NeuralNetwork.CoreObject.Variable import PlaceHolder, Constant, Trainable
from NeuralNetwork.Operation.Basic import Multiply, Add, Substract, MatMul
from NeuralNetwork.CoreObject.Session import Session

from NeuralNetwork.Operation.Gradient import gradients

import numpy as np

x = PlaceHolder( (-1, 3, 1), name = 'x' )
m = PlaceHolder( (3,3), name = 'm' )
c = PlaceHolder( (3,1), name = 'c' )

out = Add( MatMul( m, x, name="mul" ), c, name = "out" )

grad = gradients( [out], [x, m, c, out] )

mVal = np.eye( 3 )
xVal = np.arange(6).reshape(-1,3,1)
cVal = np.ones( (3,1) )

sess = Session()
print( grad )
gradVal = sess.run( grad, feedDict={x.Id:xVal, m.Id:mVal, c.Id:cVal} )

print (gradVal[0])
print( gradVal[1])
print( gradVal[2])
print( gradVal[3])