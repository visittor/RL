from Snake import Snake
from Fruit import Fruit
from Map import Map
from GameLogic import GameLogic
from Renderer import cv2Renderer
import sys

def setRenderer( snake:Snake, fruit:Fruit, renderer:cv2Renderer ):

    renderer.clear()

    renderer.setPixelValue( fruit.getPosition()[0], fruit.getPosition()[1], 255 )

    for x, y in snake.getBody():
        renderer.setPixelValue( x, y, 127 )

    head = snake.getBody()[0]

    renderer.setPixelValue( head[0], head[1], 200 )

if __name__ == '__main__':

    width = 13
    height = 13

    map_ = Map( width, height )

    snake = Snake( [(width//2, height//2)]*3 )

    fruit = Fruit( 0, 0 )

    fruit.randomPosition( (0, width), (0,height), snake.getBody() )

    gameLogic = GameLogic( snake, fruit, map_ )

    renderer = cv2Renderer( width, height, 10 )

    while not snake.isDead:
        
        setRenderer( snake, fruit, renderer )

        k = renderer.render()

        if k == ord('w'):
            k = 3
        
        elif k == ord('a'):
            k = 2
        
        elif k == ord('s'):
            k = 1

        elif k == ord('d'):
            k = 4
        
        elif k == 27:
            sys.exit( 0 )

        else:
            k = 0
        
        gameLogic.process( k )
