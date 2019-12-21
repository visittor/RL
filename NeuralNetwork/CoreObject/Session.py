from .GraphComponent import GraphComponent
from .Node import Node
from .Variable import Variable, Constant, PlaceHolder, Trainable 

from typing import List

def getGraphExecutionOrder( components: List[GraphComponent] )->List[GraphComponent]:

	orderedComponentsList = []

	def addComponent( component: GraphComponent ):

		if isinstance( component, Node ):
			for input in component.getInput():
				addComponent( input )

		if component in orderedComponentsList:
			return

		orderedComponentsList.append( component )

	for component in components:

		addComponent( component )

	return orderedComponentsList

class Session( object ):

	def __init__( self ):
		pass

	def run( self, result:List[GraphComponent], feedDict = None ):

		feedDict = feedDict if feedDict is not None else {}

		orderedComponent = getGraphExecutionOrder( result )

		for component in orderedComponent:

			if isinstance( component, Variable ):
				self._handleVariable( component, feedDict )

			elif isinstance( component, Node ):
				self._handleNode( component )

		return [ r.getOutput() for r in result ]

	def _handleVariable( self, var:Variable, feedDict ):

		if isinstance( var, PlaceHolder ):

			var.setValue( feedDict[var.Id] )

		elif isinstance( var, Constant ):

			pass

		elif isinstance( var, Trainable ):

			pass

		else:
			assert False, "Variable {}, Unknown variable type".format( var.name )

		var.setOutput()

	def _handleNode( self, node:Node ):

		inputNodes = node.getInput()

		inputDatas = [ input.getOutput() for input in inputNodes ]

		node.setOutput( node.forward( *inputDatas ) )