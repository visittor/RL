from typing import List, Tuple

class Snake( object ):

	DIRECTION_MAP = {0:(0,0), 1:(0,1), 2:(-1,0), 3:(0,-1), 4:(1,0)}

	def __init__( self, body: List[Tuple] ):

		self.__lenght = max( 1, len( body ) )

		self.__body = [(0,0)] if len(body) == 0 else body

		self.__body.append( (0,0) )

		self.__isDead = False

	def move( self, direction: int ):

		direction = self.DIRECTION_MAP[ direction ]

		head = self.__body[ 0 ]

		newHead = ( head[0] + direction[0], head[1] + direction[1] )

		if newHead == head:
			return self.__isDead

		if newHead in self.__body[:-1]:
			self.__isDead = True
			return self.__isDead

		for i in range( len(self.__body) - 1, 0, -1 ):
			self.__body[i] = (self.__body[i-1][0], self.__body[i-1][1])

		self.__body[0] = newHead

	def eat( self ):
		self.__body.append( (0,0) )

	def getBody( self ):
		return self.__body[:-1]

	@property
	def isDead( self ):
		return self.__isDead

	@property
	def lenght( self ):
		return len( self.__body )

	@property
	def head( self ):
		return self.__body[0]