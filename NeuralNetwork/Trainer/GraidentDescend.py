from NeuralNetwork.CoreObject.Graph import Graph, setDefaultGraph, getAllTrainable

from NeuralNetwork.CoreObject.GraphComponent import GraphComponent

from NeuralNetwork.Opt.Operation.Gradient import gradients

from NeuralNetwork.CoreObject.Session import Session

from typing import List
import numpy as np
import math

class GradientDescend( object ):

	def __init__( self, loss:GraphComponent, inputId:int, outputId:int, alpha:float ):

		self._loss = loss
		self._inputId = inputId
		self._outputId = outputId

		self._trainables = getAllTrainable()
		self._gradientsNode = gradients( [self._loss], self._trainables )

		self._alpha = alpha

	def trainIteration( self, x:np.ndarray, y:np.ndarray, batchSize:int, iterIdx=0 ):
		
		for i in range( 0, len(y), batchSize ):

			batchX = x[i*batchSize, (i+1)*batchSize]
			batchY = y[i*batchSize, (i+1)*batchSize]

			self.trainBatch( batchX, batchY, iterIdx=iterIdx )

	def trainBatch( self, x, y, iterIdx = 0 ):

		batchSize = len( x )

		sess = Session()

		gradsVal = sess.run( self._gradientsNode,
							feedDict={self._inputId:x, self._outputId:y} )

		self._updateWeight( self._trainables, gradsVal,
							iterIdx,
							batchSize=batchSize )

	def _updateWeight( self, trainableList:List[GraphComponent], 
						gradsVal:List[np.ndarray],
						iterIdx,
						batchSize:int = 1 ):

		for trainable, grad in zip( self._trainables, gradsVal ):

			prevVal = trainable.getOutput()

			prevVal = prevVal - ((self._alpha * grad) / batchSize)

			trainable.setValue( prevVal )

class Adam( GradientDescend ):

	def __init__( self, loss:GraphComponent, inputId:int, outputId:int, 
					alpha:float, beta1:float = 0.9, beta2:float = 0.999,
					epsilon:float = 1e-8  ):

		super( Adam, self ).__init__( loss, inputId, outputId, alpha )

		self._beta1 = beta1
		self._beta2 = beta2
		self._epsilon = epsilon

		self.initialize_M()
		self.initialize_V()

	def initialize_M( self ):

		self._m = []

		for trainable in self._trainables:
			m = np.zeros( trainable.shape )
			self._m.append( m )
		
	def initialize_V( self ):

		self._v = []

		for trainable in self._trainables:
			v = np.zeros( trainable.shape )
			self._v.append( v )

	def _updateWeight( self, trainableList:List[GraphComponent], 
						gradsVal:List[np.ndarray],
						iterIdx,
						batchSize:int = 1 ):

		for i, (trainable, grad) in enumerate(zip(trainableList, gradsVal)):

			grad_ = grad / batchSize

			self._m[i] = (self._m[i]*self._beta1) + (1-self._beta1)*grad_
			self._v[i] = (self._v[i]*self._beta2) + (1-self._beta2)*grad_

			alphaT = self._alpha * math.sqrt(1 - self._beta2**iterIdx)/(1 - self._beta1**iterIdx)

			prevVal = trainable.getOutput()

			prevVal = prevVal - (alphaT*self._m[i]) / (np.sqrt(self._v[i]) + self._epsilon)

			trainable.setValue( prevVal )
