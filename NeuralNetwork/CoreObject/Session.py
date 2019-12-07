from .GraphComponent import GraphComponent
from .Node import Node
from .Variable import Variable, Constant, PlaceHolder, Trainable 

from typing import List

def getGraphExecutionOrder( component: GraphComponent )->List[GraphComponent]:

	componentsList = []

	def addComponent( component: GraphComponent ):

		if isinstance( component, Node ):
			for input in component.getInput():
				addComponent( input )

		componentsList.append( component )

	addComponent( component )

	return componentsList

class Session( object ):

	def __init__( self ):
		pass

	def run( self, result:GraphComponent, feedDict = None ):

		feedDict = feedDict if feedDict is not None else {}

		orderedComponent = getGraphExecutionOrder( result )

		for component in orderedComponent:

			if isinstance( component, Variable ):
				self._handleVariable( component, feedDict )

			elif isinstance( component, Node ):
				self._handleNode( component )

		return result.getOutput()

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