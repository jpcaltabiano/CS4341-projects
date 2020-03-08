# This is necessary to find the main code
import collections
import heapq
import math
import random
import sys

from itertools import starmap, product

from priorityqueue import PriorityQueue

sys.path.insert(0, '../bomberman')
# Import necessary stuff
from events import Event
from entity import CharacterEntity
from world import World


def is_legal_location(state: World, location: (int, int)):
    x, y = location
    return 0 <= x < state.width() and 0 <= y < state.height() and not state.wall_at(x, y)


def neighbors(state, location):
    moves = product(tuple(range(-1, 1 + 1)), tuple(range(-1, 1 + 1)))
    locations = starmap(lambda dx, dy: (location[0] + dx, location[1] + dy), moves)
    return filter(lambda l: is_legal_location(state, l), locations)


def heuristic(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return abs(x1 - x2) + abs(y1 - y2)


def cost(state: World, location):
    return 1


def path(came_from, start, goal):
    path = [goal]

    while not path[0] == start:
        path.insert(0, came_from[path[0]])

    return path


def a_star_search(graph, start, goal):
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0

    while not frontier.empty():
        current = frontier.get()

        if current == goal:
            break

        for next in neighbors(graph, current):
            new_cost = cost_so_far[current] + cost(graph, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(goal, next)
                frontier.put(next, priority)
                came_from[next] = current

    return path(came_from, start, goal)


def bfs(state: World, start: (int, int), test):
    frontier = collections.deque()
    visited = {}
    visited[start] = True

    frontier.append((start, 0))

    while not len(frontier) == 0:
        current = frontier.popleft()

        if test(*(current[0])):
            return current[1]

        for next in neighbors(state, current[0]):
            if next not in visited:
                frontier.append((next, current[1] + 1))
                visited[next] = True
