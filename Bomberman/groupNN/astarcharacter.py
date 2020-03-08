# This is necessary to find the main code
import sys

from astarsearch import a_star_search

sys.path.insert(0, '../bomberman')
# Import necessary stuff
from entity import CharacterEntity
from colorama import Fore, Back

class AStarCharacter(CharacterEntity):

    def do(self, wrld):
        path = a_star_search(wrld, (self.x, self.y), (wrld.width() - 1, wrld.height() - 1))

        for step in path:
            self.set_cell_color(step[0], step[1], Fore.GREEN)

        dx = path[1][0] - self.x
        dy = path[1][1] - self.y
        self.move(dx, dy)
