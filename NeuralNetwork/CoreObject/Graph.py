class Graph( object ):

	def __init__( self ):
		self.placeHolder = []
		self.trainable = []
		self.constant = []

		self.node = []

_default_graph = Graph()

def getAllTrainable( ):
	return _default_graph.trainable
