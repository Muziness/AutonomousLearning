from pysc2.lib import actions
from enum import Enum

import math

_DIST_STRAIGHT = 10
_DIST_DIAGONAL = _DIST_STRAIGHT / math.sqrt(2)


# Action Set
class Actions(Enum):
    up = (actions.FUNCTIONS.Move_screen.id, "up")
    up_right = (actions.FUNCTIONS.Move_screen.id, "up_right")
    right = (actions.FUNCTIONS.Move_screen.id, "right")
    down_right = (actions.FUNCTIONS.Move_screen.id, "down_right")
    down = (actions.FUNCTIONS.Move_screen.id, "down")
    down_left = (actions.FUNCTIONS.Move_screen.id, "down_left")
    left = (actions.FUNCTIONS.Move_screen.id, "left")
    up_left = (actions.FUNCTIONS.Move_screen.id, "up_left")


def action_to_function(action, marine, screen): #, dist_x, dist_y):
    if action == Actions.up:
        return actions.FUNCTIONS.Move_screen("now", calc_target_position(marine, up, screen)) # , dist_x, dist_y))
    elif action == Actions.right:
        return actions.FUNCTIONS.Move_screen("now", calc_target_position(marine, right, screen)) # , dist_x, dist_y))
    elif action == Actions.down:
        return actions.FUNCTIONS.Move_screen("now", calc_target_position(marine, down, screen)) # , dist_x, dist_y))
    elif action == Actions.left:
        return actions.FUNCTIONS.Move_screen("now", calc_target_position(marine, left, screen)) # , dist_x, dist_y))
    elif action == Actions.up_right:
        return actions.FUNCTIONS.Move_screen("now", calc_target_position(marine, up_right, screen)) # , dist_x, dist_y))
    elif action == Actions.up_left:
        return actions.FUNCTIONS.Move_screen("now", calc_target_position(marine, up_left, screen)) # , dist_x, dist_y))
    elif action == Actions.down_right:
        return actions.FUNCTIONS.Move_screen("now", calc_target_position(marine, down_right, screen)) # , dist_x, dist_y))
    elif action == Actions.down_left:
        return actions.FUNCTIONS.Move_screen("now", calc_target_position(marine, down_left, screen)) # , dist_x, dist_y))
    """
    elif action == Actions.select:
        return actions.FUNCTIONS.select_army("select")
    """


def calc_target_position(marine, direction, screen): #, dist_x, dist_y):
    # dist_x = abs(dist_x)
    # dist_y = abs(dist_y)
    return direction(marine.x, marine.y, screen) #, dist_x, dist_y)


def up(x, y, screen): #, dist_x, dist_y):
    return check_edges(x, y - _DIST_STRAIGHT, screen)


def right(x, y, screen): #, dist_x, dist_y):
    return check_edges(x + _DIST_STRAIGHT, y, screen)


def down(x, y, screen): #, dist_x, dist_y):
    return check_edges(x, y + _DIST_STRAIGHT, screen)


def left(x, y, screen): #, dist_x, dist_y):
    return check_edges(x - _DIST_STRAIGHT, y, screen)


def up_right(x, y, screen): #, dist_x, dist_y):
    return check_edges(x + _DIST_DIAGONAL, y - _DIST_DIAGONAL, screen)


def up_left(x, y, screen): #, dist_x, dist_y):
    return check_edges(x - _DIST_DIAGONAL, y - _DIST_DIAGONAL, screen)


def down_right(x, y, screen): #, dist_x, dist_y):
    return check_edges(x + _DIST_DIAGONAL, y + _DIST_DIAGONAL, screen)


def down_left(x, y, screen): #, dist_x, dist_y):
    return check_edges(x - _DIST_DIAGONAL, y + _DIST_DIAGONAL, screen)


def check_edges(x, y, screen):

    if y < 0:
        y = 0
    elif y > screen - 1: # * 0.8 - 1:
        y = screen - 1 # * 0.8 - 1

    if x < 0:
        x = 0
    elif x > screen - 1:
        x = screen - 1

    return x, y
