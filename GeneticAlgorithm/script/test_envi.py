from SnakeGame.Enviroment import Enviroment
from SnakeGame.Renderer import cv2Renderer
import sys

if __name__ == '__main__':

	w, h = 13, 13

	renderer = cv2Renderer( w, h )
	renderer.setRefreshRate( 0 )

	env = Enviroment( w, h, renderer )

	env.startEnvi()
	k = env.process( 0, render=True )

	while not env.isAgentFail:

		if k == ord('w'):
			k = 3
		
		elif k == ord('a'):
			k = 2
		
		elif k == ord('s'):
			k = 1

		elif k == ord('d'):
			k = 4

		elif k == 27:
			env.endEnvi()
			sys.exit( 0 )

		else:
			k = 0

		k = env.process( k, render=True )