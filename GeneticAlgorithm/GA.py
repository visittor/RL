from typing import List
import numpy as np
import random

from NeuralNetwork.BaseNetwork import BaseNetwork
from SnakeGame.Enviroment import Enviroment

class GA( object ):

	def __init__( self, population:List[BaseNetwork] , envi:Enviroment, predictToInputFunc ):

		self._envi = envi

		self._population = population

		self._predictToInputFunc = predictToInputFunc

	def testOneAgent( self, agent:BaseNetwork, render:bool=False, maxIter:int=50 )->float:

		self._envi.startEnvi()
		agent.resetState()

		score = self._envi.score

		it = 0

		itUse = 0

		while not self._envi.isAgentFail and it < maxIter:

			inputVector = self._envi.getAgentPerception()

			pred = agent.predict( np.array( inputVector, dtype=np.float64 ) )

			k = self._predictToInputFunc( pred )

			self._envi.process( k, render=render )

			if score < self._envi.score:

				score = self._envi.score
				
				itUse += it
				it = 0

			it += 1
		
		speedScore = 0

		if score != 0:
			itUse_avg = itUse / score
			speedScore = ( maxIter - itUse_avg ) / maxIter

		self._envi.endEnvi()

		return score + speedScore

	def training( self, survivalRatio:float = 0.5, duplicateRation:float = 0.5,
						mutationRate:float = 0.5, mutationAmount:float = 0.5,
						seed = None )->List:

		if seed is None:
			seed = random.randrange( 0 , 1000 )

		newPop = []

		forReturn = []

		for agent in self._population:

			random.seed( seed )

			score = self.testOneAgent( agent, render=False )

			newPop.append( (score, agent) )

			forReturn.append( (score, agent.getFlattenWeight().copy()) )

		random.seed()

		newChro = self.getNewPopulation(newPop, survivalRatio=survivalRatio, duplicateRation=duplicateRation,
												mutationRate=mutationRate, mutationAmount=mutationAmount 
										)
		for agent, chro in zip(self._population, newChro ):

			agent.setWeight( chro )
			agent.resetState()

		return forReturn

	def getNewPopulation( self, population:List, survivalRatio:float = 0.5, duplicateRation:float = 0.5,
							mutationRate:float = 0.5, mutationAmount:float = 0.5 ):

		nPopulation = len(population)

		population = sorted( population, key=lambda x : x[0], reverse=True )
		survivals = [ p[1].getFlattenWeight().copy() for p in population[:int(nPopulation * survivalRatio)] ]
		
		weights = [ p[0] for p in population[:int(nPopulation * survivalRatio)] ]
		sumWeight = sum( weights )
		weights = [ w / sumWeight for w in weights ]

		nDuplicate = int((nPopulation - len(survivals)) * duplicateRation)
		nCrossOver = nPopulation - (len(survivals) + nDuplicate)

		duplicatePop = self.duplicate( survivals, weights, nDuplicate )
		crossOverPop = self.crossingOver( survivals, weights, nCrossOver )

		childPop = duplicatePop + crossOverPop

		childPop =  self.mutate( childPop, mutationRate, mutationAmount )

		newGeneration = survivals + childPop

		assert nPopulation == len( newGeneration )

		return newGeneration

	def duplicate( self, chromosomes:List[np.ndarray], weights:List[float], nDup:int ):

		chosenPop = np.random.choice( len(chromosomes), p=weights, size = nDup )

		return [ chromosomes[i].copy() for i in chosenPop ]

	def mutate( self, chromosomes:List[np.ndarray], mutateRate:float, mutateAmount:float ):

		mutatePop = []

		for chro in chromosomes:

			randomState = np.random.normal( scale=mutateAmount, size = chro.shape )
			mask = np.random.rand( *randomState.shape ) < mutateRate

			mutatePop.append( np.multiply( randomState, mask ) )

		return mutatePop

	def crossingOver( self, chromosomes:List[np.ndarray], weights:List[float], nChild:int ):

		childList = []

		for _ in range( nChild ):

			chosen = random.choices( chromosomes, weights=weights, k = 2 )
			chosen = np.random.choice( len(chromosomes), p=weights, size = 2 )

			father = chromosomes[chosen[0]]
			mother = chromosomes[chosen[1]]

			nWeight = len( father )

			child  = np.hstack( [father[:int(nWeight/2)], mother[int(nWeight/2):] ] )

			assert child.shape == father.shape

			childList.append( child )

		return childList
