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

    def do(self, wrld):
        features = self.get_features(wrld)
        weights = [2, 2, 2, 2]
        self.qlearn(features, weights)
        # print(features)
        # features.reshape(1, -1)
        # print(features)
        # features = StandardScaler().fit_transform(features)
        # print(features)

    def qlearn(self, features, weights):
        state_val = (features * weights).sum()

        # assuming this is a possible next move to an empty spot
        # get features of this move, calculate value with current weights?????
        next_val = 100
        reward = 10
        lr = 0.2
        delta = (reward + next_val) - state_val
        print(weights)
        for i, _ in enumerate(weights):
            weights[i] = weights[i] + (lr * delta * features[i])
        print(weights)

    def load_weights(self):
        # will load from somewhere else during training
        return 0

    # save new weights to file after every move
    def save_weights(self):
        return 0

    # features (in order) [mdist to mon1, mdist to mon2, mdist to bomb, mdist to exit]
    def get_features(self, wrld):
        mdist = self.monster_dist(wrld)
        edist = self.exit_dist(wrld)
        bdist = self.bomb_dist(wrld)
        return np.array([mdist[0], mdist[1], bdist, edist])

    def monster_dist(self, wrld):
        dlist = []
        for m in wrld.monsters.values():
            # Manhattan distance
            dlist.append(abs(self.x - m[0].x) + abs(self.y - m[0].y))
            # Euclidean distance
        return dlist

    def bomb_dist(self, wrld):
        # TODO: just get the closest bomb, currently working under
        # assumption only 1 bomb exists
        if (len(wrld.bombs) == 0): return 0
        return (abs(self.x - wrld.bombs[0].x) + abs(self.y - wrld.bombs[0].y))

    def exit_dist(self, wrld):
        return (abs(self.x - wrld.width()-1) + abs(self.y - wrld.height()-1))

    

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