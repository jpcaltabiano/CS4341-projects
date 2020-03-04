# This is necessary to find the main code
import math
import sys
sys.path.insert(0, '../bomberman')
# Import necessary stuff
from entity import CharacterEntity
from colorama import Fore, Back

import heapq
from itertools import product, starmap

from sklearn.preprocessing import StandardScaler
import numpy as np

class ApproxQCharacter(CharacterEntity):

    def __init__(self, name, avatar, x, y):
        super().__init__(name, avatar, x, y)
        self.ws = [2, 2, 2, 2]

    def do(self, wrld):
        # weights = [2, 2, 2, 2]
        next_move = self.choose_move(wrld, self.ws)

        print(next_move)
        dx, dy = next_move[0][0] - self.x, next_move[0][1] - self.y
        self.move(dx, dy)
 
        state_fts = self.get_features(wrld, (self.x, self.y))
        state_val = (state_fts * self.ws).sum()

        reward = -1 if (dx, dy) == (0, 0) else 2

        self.ws = self.update_weights(self.ws, state_val, next_move, reward, 0.5)

        # features.reshape(1, -1)
        # features = StandardScaler().fit_transform(features)

    def choose_move(self, wrld, weights):

        nbors = self.neighbors(wrld, (self.x, self.y))
        next_move = (0, -math.inf, 0)
        # selecting the neighbor cell with highest evaluation
        # TODO: Do only if epsilon-greedy chooses exploit over random move
        #   otherwise just find the evaluation of the chosen neighbor
        for n in nbors:
            # n == (self.x, self.y) or
            # if n == (0, 0) or wrld.wall_at(self.x+n[0], self.y+n[1]):
            #     continue
            next_fts = self.get_features(wrld, (n[0], n[1]))
            next_val = (next_fts * weights).sum()
            next_move = (n, next_val, next_fts) if next_val > next_move[1] else next_move

        # updating weights should only happen after the move has been made?
        # Where do the reward values come from?? How will we know to update weights
        # if we die and the execution ends???
        return next_move

    def update_weights(self, weights, state_val, next_move, reward, lr):
        delta = (reward + next_move[1]) - state_val
        print(weights)
        for i, _ in enumerate(weights):
            weights[i] = weights[i] + (lr * delta * next_move[2][i])
        print(weights)
        return weights

    def load_weights(self):
        # will load from somewhere else during training
        return 0

    # save new weights to file after every move
    def save_weights(self):
        return 0

    # features (in order) [mdist to mon1, mdist to mon2, mdist to bomb, mdist to exit]
    def get_features(self, wrld, loc):
        mdist = self.monster_dist(wrld, loc)
        edist = self.exit_dist(wrld, loc)
        bdist = self.bomb_dist(wrld, loc)
        features = np.array([mdist[0], mdist[1], bdist, edist])
        normalized = features / np.linalg.norm(features)
        return normalized


    def monster_dist(self, wrld, loc):
        dlist = []
        for m in wrld.monsters.values():
            # Manhattan distance
            dlist.append(abs(loc[0] - m[0].x) + abs(loc[1] - m[0].y))
            # Euclidean distance
        return dlist

    def bomb_dist(self, wrld, loc):
        # TODO: just get the closest bomb, currently working under
        # assumption only 1 bomb exists
        if (len(wrld.bombs) == 0): return 0
        return (abs(loc[0] - wrld.bombs[0].x) + abs(loc[1] - wrld.bombs[0].y))

    def exit_dist(self, wrld, loc):
        return (abs(loc[0] - wrld.width()-1) + abs(loc[1] - wrld.height()-1))

    # TODO: Don't return neighbors you cant move to (walls)
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