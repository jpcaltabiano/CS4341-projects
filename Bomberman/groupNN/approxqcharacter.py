# This is necessary to find the main code
import math
import sys
sys.path.insert(0, '../bomberman')
# Import necessary stuff
from events import Event
from entity import CharacterEntity
from colorama import Fore, Back

import heapq
from itertools import product, starmap

from sklearn.preprocessing import StandardScaler
import numpy as np
import random

#TODO: Are there going to be more than one non-monster player chars at a time?
class ApproxQCharacter(CharacterEntity):

    def __init__(self, name, avatar, x, y):
        super().__init__(name, avatar, x, y)
        self.ws = [0, 0, 0, 0]
        self.visited = []

    def do(self, wrld):
        state_fts = self.get_features(wrld, (self.x, self.y), False)
        state_val = (state_fts * self.ws).sum()

        #TODO: Need to include not moving and placing bomb as a valid action
        if random.random() > 0.4: 
            next_move = self.choose_best_move(wrld)
        else:
            next_move = self.choose_random_move(wrld)

        #TODO: Can you cal place_bomb(), then move() in the same call to do()?
        if next_move[3]: self.place_bomb()
        dx, dy = next_move[0][0] - self.x, next_move[0][1] - self.y
        self.move(dx, dy)
        self.visited.append(next_move[0])
 
        reward = self.get_reward(wrld, next_move)
        lr = 0.1

        self.ws = self.update_weights(self.ws, state_val, next_move, reward, lr)

    def get_reward(self, wrld, move):
        #TODO: How do we know if the agent tries to move to a wall or past boundary?
        rw = 0
        rw = -0.1 if move in self.visited else 0.1
        if move[3] and [0, 1, 2] not in wrld.events: rw = -0.7

        for e in wrld.events:
            if e == Event.BOMB_HIT_WALL: rw = 0.3
            if e == Event.BOMB_HIT_MONSTER: rw = 0.7
            if e == Event.BOMB_HIT_CHARACTER: rw = -0.9
            if e == Event.CHARACTER_KILLED_BY_MONSTER: rw = -1
            if e == Event.CHARACTER_FOUND_EXIT: rw = 1

        print("qchar reward: ", rw)
        return rw
        
    def choose_random_move(self, wrld):
        nbors = self.neighbors(wrld, (self.x, self.y))
        move = list(nbors)[random.randint(0, len(list(nbors)))]
        fts = self.get_features(wrld, (move[0], move[1]), False)
        val = (fts * self.ws).sum()
        b = True if random.random() > 0.5 else False
        return (move, val, fts, b)

    def choose_best_move(self, wrld):
        nbors = self.neighbors(wrld, (self.x, self.y))
        next_move = (0, -math.inf, 0, False)
        # selecting the neighbor cell with highest evaluation
        for n in nbors:
            next_fts = self.get_features(wrld, (n[0], n[1]), False)
            next_val = (next_fts * self.ws).sum()
            next_move = (n, next_val, next_fts, False) if next_val > next_move[1] else next_move

        bomb_fts = self.get_features(wrld, (self.x, self.y), True)
        bomb_val = (bomb_fts * self.ws).sum()
        next_move = ((self.x, self.y), bomb_val, bomb_fts, True) if bomb_val > next_move[1] else next_move
        return next_move

    def update_weights(self, weights, state_val, next_move, reward, lr):
        delta = (reward + next_move[1]) - state_val
        for i, _ in enumerate(weights):
            weights[i] = weights[i] + (lr * delta * next_move[2][i])
        return weights

    def load_weights(self):
        # will load from somewhere else during training
        return 0

    # save new weights to file after every move
    def save_weights(self):
        return 0

    # features (in order) [mdist to mon1, mdist to mon2, mdist to bomb, mdist to exit]
    def get_features(self, wrld, loc, bomb):
        mdist = self.monster_dist(wrld, loc)
        edist = self.exit_dist(wrld, loc)
        bdist = self.bomb_dist(wrld, loc)
        place_bomb = 1 if bomb else 0
        features = np.array([mdist, bdist, edist, place_bomb])
        normalized = features / np.linalg.norm(features)
        return normalized


    def monster_dist(self, wrld, loc):
        # Get the distance to the closest monster or 25
        if not wrld.monsters:
            return 25

        return min(map(lambda monster: abs(loc[0] - monster[0].x) + abs(loc[1] - monster[0].y), wrld.monsters.values()))

    def bomb_dist(self, wrld, loc):
        # Get the distance to the closest bomb or 25
        if not wrld.bombs:
            return 25

        return min(map(lambda bomb: abs(loc[0] - bomb.x) + abs(loc[1] - bomb.y), wrld.bombs.values()))

    def exit_dist(self, wrld, loc):
        return (abs(loc[0] - wrld.width()-1) + abs(loc[1] - wrld.height()-1))

    def neighbors(self, wrld, location, distance=1):
        x, y = location
        cells = starmap(lambda dx, dy: (x + dx, y + dy), product(tuple(range(-distance, distance+1)), tuple(range(-distance, distance+1))))
        return filter(lambda loc: 0 <= loc[0] < wrld.width() and 0 <= loc[1] < wrld.height() and not wrld.wall_at(loc[0], loc[1]), cells)

    '''

    Features to use:
        dist to nearest monster
        dist to nearest bomb
        dist to nearest exit
        direction of monster travel
        ?

        normalize all features

        can agent tell if monster is stupid or self preserving?

    '''