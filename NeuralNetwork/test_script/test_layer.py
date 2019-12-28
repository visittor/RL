from NeuralNetwork.Opt.Layer.Dense import Dense
from NeuralNetwork.Opt.Operation.Basic import Reshape, Transpose
from NeuralNetwork.Opt.Operation.Activation import Relu, SoftMax, Sigmoid, Tanh
from NeuralNetwork.CoreObject.Variable import PlaceHolder
from NeuralNetwork.Opt.Operation.LossFunc import LogLoss
from NeuralNetwork.Opt.Operation.Gradient import gradients
from NeuralNetwork.CoreObject.Session import Session
from NeuralNetwork.CoreObject.Graph import getAllTrainable

import numpy as np

inp = PlaceHolder( (-1,2,1), name="input" )
gt = PlaceHolder( ( -1, 1 ), name="ground_truth" )

dense1 = Dense( 2, Relu, 2, name='dense1' )(inp) #	(-1, 2, 1)
dense2 = Dense( 2, Relu, 2, name='dense2' )(dense1) # (-1, 2, 1)
dense3 = Dense( 2, Relu, 2, name='dense3' )(dense2) # (-1, 2, 1)
dense4 = Dense( 2, Relu, 2, name='dense4' )(dense3) # (-1, 2, 1)
out = Dense( 1, Sigmoid, 2, name='out' )(dense4) # (-1, 2, 1)
outRe = Reshape( out, (-1,1) )
print( out.shape )
loss = LogLoss( outRe, gt, name = 'loss' )
print( loss.shape )

# inpVal = np.random.randn( 2, 500 )
gtVal = np.zeros( (4, 1) )
inpVal = np.array( [[1,1], [-1,1], [-1,-1], [1,-1]] ).reshape( -1, 2, 1 )

grads = gradients( [loss], getAllTrainable() + [outRe] + [out], grad_y=1 )

# for g in grads:
# 	print( g.shape)

idx = np.where( inpVal[:,0] * inpVal[:,1] > 0 )
gtVal[idx, 0 ] = 1
print( gtVal )
sess = Session()

trainables = getAllTrainable()

alpha = 0.01

for i in range( 1000 ):
	print('-----')
	gradsVal = sess.run( grads + [outRe] , feedDict={inp.Id:inpVal, gt.Id:gtVal} )
	print( loss.getOutput() )
	# print( gradsVal[-1][:1], gradsVal[-1][:1].sum() )
	# print( gradsVal[-3][:1] )
	# print( gradsVal[-2][:,:1])
	# print( gtVal[:1] )
	#	update weight
	for trainable, grad in zip(trainables, gradsVal[:-3]):
		prevVal = trainable.getOutput()
		# print( grad.shape, prevVal.shape )
		prevVal = prevVal - (alpha * (grad) / 4)
		trainable.setValue( prevVal )