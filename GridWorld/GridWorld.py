import numpy as np
from enum import Enum
from Helpers import matprint


MOVEREWARD = -1
GOALREWARD = 0

class ActionEnum(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4

class GridWorld(object):

    def __init__(self, height, width, blocked_fraction=0):
        print(f"Initiating {height}x{width} GridWorld, with {blocked_fraction*100}% blocked.\n")

        self.height = height
        self.width = width
        self.grid = np.zeros((height, width))

        self.available_actions = [1, 2, 3, 4]  # 1: Left, 2: Up, 3: Right, 4: Down

        self.current_pos = [0, 0]

    def action(self, ac):

        reward = GOALREWARD
        oob = False

        if self.current_pos[1] == 0 and ac == ActionEnum.LEFT:
            oob = True
        elif self.current_pos[1] == self.width-1 and ac == ActionEnum.RIGHT:
            oob = True
        elif self.current_pos[0] == 0 and ac == ActionEnum.UP:
            oob = True
        elif self.current_pos[0] == self.height-1 and ac == ActionEnum.DOWN:
            oob = True

        if not oob:
            if ac == ActionEnum.LEFT:
                self.current_pos[1] -= 1
            elif ac == ActionEnum.RIGHT:
                self.current_pos[1] += 1
            elif ac == ActionEnum.UP:
                self.current_pos[0] -= 1
            elif ac == ActionEnum.DOWN:
                self.current_pos[0] += 1

        #state = self.grid.copy()
        #state[self.current_pos[0], self.current_pos[1]] = 1

        return self.current_pos, reward

    def visualize(self):
        vis_grid = self.grid.copy()
        vis_grid[self.current_pos[0], self.current_pos[1]] = 1
        matprint(vis_grid)


if __name__ == "__main__":
    e = GridWorld(4, 4)

    e.visualize()
