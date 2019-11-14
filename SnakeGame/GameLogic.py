from .Snake import Snake
from .Fruit import Fruit
from .Map import Map

class GameLogic( object ):

	def __init__(self, snake: Snake, fruit: Fruit, map_: Map ):

		self.__snake = snake
		self.__fruit = fruit
		self.__map = map_

	def process( self, k:int ):

		head = self.__snake.getBody()[0]
		direction = Snake.DIRECTION_MAP[k]

		newHead = (head[0] + direction[0], head[1] + direction[1])

		if 0 > newHead[0] or newHead[0]>= self.__map.width:
			return
		
		if 0 > newHead[1] or newHead[1]>= self.__map.height:
			return

		self.__snake.move( k )

		isDead = self.__snake.isDead

		if isDead:
			return isDead

		if self.__fruit.getPosition() in self.__snake.getBody():

			self.__snake.eat()

			self.__fruit.randomPosition( (0,self.__map.width), (0,self.__map.height), invalidPos=self.__snake.getBody() )
		
		return isDead