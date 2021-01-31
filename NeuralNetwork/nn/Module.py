from abc import abstractclassmethod, ABCMeta

from ..Graph.Tensor import Tensor

class Module( metaclass = ABCMeta ):

	def __init__( self ):

		self._parameters = {}
		self._modules = {}

	def getAllParameters( self ):

		parameters = []

		for param in self._parameters.values():

			parameters.append( param )

		for module in self._modules.values():

			parameters.extend( module.getAllParameters() )

		return parameters

	def __setattr__( self, name : str, var ):

		if isinstance( var, Tensor ) and var.is_leaf:

			if not hasattr( self, "_parameters" ):
				raise RuntimeError( "Used Module without calling \'__init__\'." )

			if name.endswith( "_par" ):

				parName = name.replace( "_par", "" )

				self._parameters[ parName ] = var

				super( Module, self ).__setattr__( name, var )

				return
		
		if isinstance( var, Module ):

			if not hasattr( self, "_modules" ):
				raise RuntimeError( "Used Module without calling \'__init__\'." )

			if name.endswith( "_module" ):

				parName = name.replace( "_module", "" )

				self._modules[ parName ] = var

				super( Module, self ).__setattr__( name, var )

				return

		super( Module, self ).__setattr__( name, var )

	@abstractclassmethod
	def forward( self, *inputs ):
		raise NotImplementedError