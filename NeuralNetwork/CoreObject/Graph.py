class Graph( object ):
	'''
		Container of GraphComponents. Conveniently for accessing components by its type.
	'''
	def __init__( self ):
		self.placeHolder = []
		self.trainable = []
		self.constant = []

		self.node = []

#	There can be more than one graphs per program
#	So, we can switch it back and forth.
#	It will use it like a name space
_graph_lookup = { "DEFAULT":Graph() }

_default_graph = _graph_lookup[ "DEFAULT" ]

def getAllTrainable( ):
	return _default_graph.trainable

def getPlaceHolderIdByName( name:str ):

	for placeholder in _default_graph.placeHolder:
		if placeholder.name == name:
			return placeholder

def setDefaultGraph( name:str ):
	assert name in _graph_lookup, "Graph {} is not exist".format( name )

	_default_graph = _graph_lookup[ name ]

def addGraph( graph:Graph, name:str ):
	assert name not in _graph_lookup, "Duplicate graph name"

	_graph_lookup[ name ] = Graph

def getAllGraphName()->str:
	return _graph_lookup.keys()

def getGraph( name:str )->Graph:
	return _graph_lookup[name]