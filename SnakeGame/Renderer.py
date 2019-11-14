import cv2
import numpy as np

class AbstractRenderer( object ):

	def __init__( self, width: int, height: int ):

		assert width > 0
		assert height > 0

		self._width = width
		self._height = height

		self._pixelVal = np.zeros( (self._height, self._width ), dtype = np.uint8 )

	def clear( self ):
		self._pixelVal[:,:] = 0

	def setPixelValue( self, x:int, y:int, value:int ):
		assert x >= 0 and x < self._width, x
		assert y >= 0 and y < self._height, y

		self._pixelVal[y, x] = value

	def render( self ):
		raise NotImplementedError

	def tearDown( self ):
		pass

class cv2Renderer( AbstractRenderer ):

	def __init__( self, width:int, height:int, pixPerTile:int = 10 ):
		super( cv2Renderer, self ).__init__( width, height )

		assert pixPerTile > 0

		self._image = np.zeros( (width*pixPerTile, height*pixPerTile), dtype = np.uint8 )

		self._pixPerTile = pixPerTile

		self._refreshRate = 1

	def setRefreshRate( self, refreshRate: int ):

		self._refreshRate = refreshRate

	def render( self ):
		self._image = cv2.resize(   self._pixelVal.copy(), None, 
									fx = self._pixPerTile, fy = self._pixPerTile, 
									interpolation = cv2.INTER_NEAREST )

		cv2.imshow( 'Image', self._image )
		return cv2.waitKey( self._refreshRate )

	def tearDown( self ):
		cv2.destroyAllWindows()

	def __getPixBound( self, x, y ):

		x2 = x + self._pixPerTile
		y2 = y + self._pixPerTile

		return x, y, x2, y2
