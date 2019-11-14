import numpy as np

class BaseNetwork( object ):

	def __init__( self ):
		raise NotImplementedError

	def initWeight( self ):
		raise NotImplementedError

	def getFlattenWeight( self )->np.ndarray:
		raise NotImplementedError

	def setWeight( self, flatWeight:np.ndarray )->np.ndarray:
		raise NotImplementedError

	def predict( self, x:np.ndarray )->np.ndarray:
		raise NotImplementedError

	def resetState( self ):
		pass

	def nWeight( self )->int:
		raise NotImplementedError