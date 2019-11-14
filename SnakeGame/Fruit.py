import random
from typing import Tuple

class Fruit( object ):

    def __init__( self, positionX: int, positionY: int ):

        self.__positionX = positionX
        self.__positionY = positionY

    def randomPosition( self, xlim: Tuple[int], ylim: Tuple[int], invalidPos = None ):
        
        invalidPos = [] if invalidPos is None else invalidPos

        self.__positionX = random.randrange( xlim[0], xlim[1] )

        self.__positionY = random.randrange( ylim[0], ylim[1] )

        while (self.__positionX, self.__positionY) in invalidPos:

            self.__positionX = random.randrange( xlim[0], xlim[1] )

            self.__positionY = random.randrange( ylim[0], ylim[1] )

    def getPosition( self ):

        return self.__positionX, self.__positionY