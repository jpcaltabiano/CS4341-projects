# This is necessary to find the main code
import math
import sys
sys.path.insert(0, '../bomberman')
# Import necessary stuff
from entity import CharacterEntity
from colorama import Fore, Back

import heapq
from itertools import product, starmap


class AStarCharacter(CharacterEntity):

    def do(self, wrld):
        self.tiles = {}
        path = self.a_star_search(wrld, (self.x, self.y), (wrld.width() - 1, wrld.height() - 1))

        for step in path:
            self.set_cell_color(step[0], step[1], Fore.GREEN)

        dx = path[1][0] - self.x
        dy = path[1][1] - self.y
        self.move(dx, dy)

    @staticmethod
    def path(came_from, start, goal):
        path = [goal]

        while not path[0] == start:
            path.insert(0, came_from[path[0]])

        return path


    @staticmethod
    def neighbors(wrld, location, distance=1):
        x, y = location
        cells = starmap(lambda dx, dy: (x + dx, y + dy), product(tuple(range(-distance, distance+1)), tuple(range(-distance, distance+1))))
        return filter(lambda loc: 0 <= loc[0] < wrld.width() and 0 <= loc[1] < wrld.height() and not wrld.wall_at(loc[0], loc[1]), cells)

    def cost(self, wrld, location):
        nearest_monter = math.inf
        for monster in wrld.monsters.values():
            nearest_monter = min(nearest_monter, AStarCharacter.heuristic(location, (monster[0].x, monster[0].y)))

        cost = 1 + max(0, (10 - nearest_monter))

        # for dist in range(1, 4):
        #     for n in AStarCharacter.neighbors(wrld, location, dist):
        #         x, y = n
        #         if wrld.monsters_at(x, y):
        #             cost += 1
        #             self.set_cell_color(x, y, Fore.RED)

        return cost

    @staticmethod
    def heuristic(a, b):
        (x1, y1) = a
        (x2, y2) = b
        return abs(x1 - x2) + abs(y1 - y2)

    def a_star_search(self, graph, start, goal):
        frontier = PriorityQueue()
        frontier.put(start, 0)
        came_from = {}
        cost_so_far = {}
        came_from[start] = None
        cost_so_far[start] = 0
        
        while not frontier.empty():
            current = frontier.get()
            
            # if current == goal:
            #     break
            
            for next in AStarCharacter.neighbors(graph, current):
                new_cost = cost_so_far[current] + self.cost(graph, next)
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + AStarCharacter.heuristic(goal, next)
                    frontier.put(next, priority)
                    came_from[next] = current

        return AStarCharacter.path(came_from, start, goal)
        # return came_from, cost_so_far


class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]