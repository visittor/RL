
class Map( object ):

    def __init__( self, width: int, height: int ):

        self.__width = width
        self.__height = height

    @property
    def width( self ):

        return self.__width
    
    @property
    def height( self ):

        return self.__height