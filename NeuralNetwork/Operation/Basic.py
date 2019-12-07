from ..CoreObject.Node import Node
from ..CoreObject.GraphComponent import GraphComponent

import numpy as np

from typing import List

class Multiply( Node ):
	
	def __init__( self, x:GraphComponent, y:GraphComponent,
				**kwargs ):
		super( Multiply, self ).__init__( inputs=[x, y], **kwargs )

	def forward( self, x:np.ndarray, y:np.ndarray )->np.ndarray:
		return np.matmul( x, y )

	def backward( self, dL:np.ndarray, prevX:np.ndarray, 
		prevY:np.ndarray )->List[np.ndarray]:

		dX = np.matmul( dL, prevY.T )
		dY = np.matmul( prevX.T, dL )

		return [dX, dY]

class Divide( Node ):

	def __init__( self, x:GraphComponent, y:GraphComponent,
				**kwargs ):
		super( Divide, self ).__init__( inputs=[x, y], **kwargs )

	def forward( self, x:np.ndarray, y:np.ndarray )->np.ndarray:
		return np.divide( x, y )

	def backward( self, dL:np.ndarray, prevX:np.ndarray, 
		prevY:np.ndarray )->List[np.ndarray]:

		dX = np.matmul( 1, prevY )
		dY = -1 * np.matmul( prevX, np.power(prevY, 2) )

		return [dX, dY]

class Add( Node ):

	def __init__( self, x:GraphComponent, y:GraphComponent,
				**kwargs ):
		super( Add, self ).__init__( inputs=[x, y], **kwargs )

	def forward( self, x:np.ndarray, y:np.ndarray )->np.ndarray:
		return x + y

	def backward( self, dL:np.ndarray, prevX:np.ndarray, 
		prevY:np.ndarray )->List[np.ndarray]:

		dX = dL.copy()
		dY = dL.copy()

		return [dX, dY]

class Substract( Node ):

	def __init__( self, x:GraphComponent, y:GraphComponent,
				**kwargs ):
		super( Substract, self ).__init__( inputs=[x, y], **kwargs )

	def forward( self, x:np.ndarray, y:np.ndarray )->np.ndarray:
		return x - y

	def backward( self, dL:np.ndarray, prevX:np.ndarray, 
		prevY:np.ndarray )->List[np.ndarray]:

		dX = dL.copy()
		dY = -1*dL

		return [dX, dY]

class Max( Node ):

	def __init__( self, x:GraphComponent, y:GraphComponent,
				**kwargs ):
		super( Max, self ).__init__( inputs=[x, y], **kwargs )

	def forward( self, x:np.ndarray, y:np.ndarray )->np.ndarray:
		return np.maximum( x, y )

	def backward( self, dL:np.ndarray, prevX:np.ndarray, 
		prevY:np.ndarray )->List[np.ndarray]:

		dX = dL * (prevX >= prevY)
		dY = dL * (prevX <= prevY)

		return [dX, dY]

class Exponent( Node ):

	def __init__( self, x:GraphComponent, y:GraphComponent,
				**kwargs ):
		super( Exponent, self ).__init__( inputs=[x, y], **kwargs )

	def forward( self, x:np.ndarray, y:np.ndarray )->np.ndarray:
		return np.power( x, y )

	def backward( self, dL:np.ndarray, prevX:np.ndarray, 
		prevY:np.ndarray )->List[np.ndarray]:

		dX = np.multiply( prevY, np.power( prevX, prevY - 1 ) )
		dX = np.multiply( dL, dX )

		dY = np.multiply(np.power( prevX, prevY ), np.log( prevX ) )
		dY = np.multiply( dL, dY )

		return [dX, dY]

class Sigmoid( Node ):

	def __init__( self, x:GraphComponent, **kwargs ):
		super( Sigmoid, self ).__init__( inputs=[x], **kwargs )

	def forward( self, x:np.ndarray )->np.ndarray:
		return np.power( np.e, x )

	def backward( self, dL:np.ndarray, prevX:np.ndarray )->List[np.ndarray]:
		
		sig = np.power( np.e, prevX )

		dX = np.multiply( sig, 1 - sig )

		return [dX]

class Tanh( Node ):

	def __init__( self, x:GraphComponent, **kwargs ):
		super( Tanh, self ).__init__( inputs=[x], **kwargs )

	def forward( self, x:np.ndarray )->np.ndarray:
		return np.tanh( x )

	def backward( self, dL:np.ndarray, prevX:np.ndarray )->List[np.ndarray]:

		tanh = np.tanh( prevX )

		dX = 1 - np.power( tanh, 2 )

		return [dX]