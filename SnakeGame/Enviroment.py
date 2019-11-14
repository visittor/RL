from .Snake import Snake
from .Fruit import Fruit
from .Map import Map
from .GameLogic import GameLogic
from .Renderer import AbstractRenderer

from typing import List

_HUGE = 1e+6

def block( func ):

	def blockingFunc( self, *arg, **kwargs ):

		assert self._isStartEnvi

		return func( self, *arg, **kwargs )
	
	return blockingFunc

class Enviroment( object ):

	def __init__( self, width: int, height: int, renderer:AbstractRenderer ):

		self._width = width
		self._height = height

		self._isStartEnvi = False

		self._renderer = renderer

	def startEnvi( self ):

		self._map_ = Map( self._width, self._height )

		self._snake = Snake( [(self._width//2, self._height//2)]*3 )

		self._fruit = Fruit( 0, 0 )

		self._fruit.randomPosition( (0, self._width), (0,self._height), self._snake.getBody() )

		self._gameLogic = GameLogic( self._snake, self._fruit, self._map_ )

		self._isStartEnvi = True
	
	def endEnvi( self ):

		self._isStartEnvi = False
		self._renderer.tearDown()

	@block
	def getAgentPerception( self )->List[int]:

		directions = [(0,-1), (1,-1), (1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1)]

		outputVector = []

		fruitPos = self._fruit.getPosition()

		deltaFruit = ( fruitPos[0] - self._snake.head[0], fruitPos[1] - self._snake.head[1] )

		body = self._snake.getBody()

		for direction in directions:
			outputDirection = [0, 0, 0]

			pos = list(self._snake.head)

			delX = pos[0] + 1 if direction[0] < 0 else self._width - pos[0]
			delX = delX if direction[0] != 0 else _HUGE

			delY = pos[1] + 1 if direction[1] < 0 else self._height - pos[1]
			delY = delY if direction[1] != 0 else _HUGE

			outputDirection[2] = min( delX, delY )

			if (deltaFruit[0] == 0 and direction[0] == 0) or deltaFruit[0] * direction[0] > 0:

				if (deltaFruit[1] == 0 and direction[1] == 0) or deltaFruit[1] * direction[1] > 0:

					if abs(deltaFruit[0]) == abs(deltaFruit[1]) or deltaFruit[0]*deltaFruit[1] == 0:

						outputDirection[0] = max( abs(deltaFruit[0]), abs(deltaFruit[1]) )

			distance = 0

			while 0 <= pos[0] < self._width and 0 <= pos[1] < self._height:

				pos[0] += direction[0]
				pos[1] += direction[1]

				distance += 1

				if tuple(pos) in body:
					outputDirection[1] = distance
					break

			outputVector.extend( outputDirection )

		return outputVector

	@staticmethod
	def setRenderer( snake:Snake, fruit:Fruit, renderer:AbstractRenderer ):

		renderer.clear()

		renderer.setPixelValue( fruit.getPosition()[0], fruit.getPosition()[1], 255 )

		for x, y in snake.getBody():
			renderer.setPixelValue( x, y, 127 )

		head = snake.getBody()[0]

		renderer.setPixelValue( head[0], head[1], 200 )

	@block
	def process( self, k, render = False ):
		
		self._gameLogic.process( k )
		if render:

			self.setRenderer( self._snake, self._fruit, self._renderer )
			return self._renderer.render()

	@property
	@block
	def score( self ):
		return self._snake.lenght - 3

	@property
	@block
	def isAgentFail( self ):
		return self._snake.isDead