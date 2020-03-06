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
        self.ws = [0, 0, 0]
        self.visited = []
        self.exitSuccess = 0
        self.monsterKilled = 0
        self.wallExploded = 0

    def do(self, wrld):
        state_fts = self.get_features(wrld, (self.x, self.y))
        state_val = (state_fts * self.ws).sum()

        #TODO: Need to include not moving and placing bomb as a valid action
        if random.random() > 0.2: 
            next_move = self.choose_best_move(wrld)
        else:
            next_move = self.choose_random_move(wrld)

        if next_move[0][2]:
            self.place_bomb()
        self.move(next_move[0][0], next_move[0][1])
        
        self.visited.append((self.x, self.y))
 
        reward = self.get_reward(wrld, next_move[0])
        lr = 0.01

        self.ws = self.update_weights(self.ws, state_val, next_move, reward, lr)

    def get_reward(self, wrld, action):
        rw = 0

        if (action[0], action[1]) in self.visited:
            rw += -0.04
        else:
            rw += 0.04

        for e in wrld.events:
            if e == Event.BOMB_HIT_WALL:
                rw = 0.3
                self.wallExploded += 1
            if e == Event.BOMB_HIT_MONSTER:
                rw = 0.7
                self.monsterKilled += 1
            if e == Event.BOMB_HIT_CHARACTER: rw = -0.9
            if e == Event.CHARACTER_KILLED_BY_MONSTER: rw = -1
            if e == Event.CHARACTER_FOUND_EXIT:
                rw = 1
                self.exitSuccess += 1

        print("qchar reward: ", rw)
        return rw
        
    def choose_random_move(self, wrld):
        all_actions = list(self.get_possible_actions(wrld, (self.x, self.y)))
        return self.get_move_from_action(wrld, random.choice(all_actions))

    def choose_best_move(self, wrld):
        all_actions = self.get_possible_actions(wrld, (self.x, self.y))
        moves = map(lambda action: self.get_move_from_action(wrld, action), all_actions)
        return max(moves, key=lambda move: move[1])

    def get_move_from_action(self, wrld, action):
        fts = self.get_features(wrld, (self.x + action[0], self.y + action[1]))
        val = (fts * self.ws).sum()
        return (action, val, fts)

    def get_possible_actions(self, wrld, loc):
        """
        Get the list of possible actions with the given world state.
        
        An action is a dx, dy, and place bomb tuple.
        """
        x, y = loc

        all_actions = product(tuple(range(-1, 1+1)), tuple(range(-1, 1+1)), (True, False))

        return filter(lambda action: 0 <= action[0] + x < wrld.width() and 0 <= action[1] + y < wrld.height() and not wrld.wall_at(action[0] + x, action[1] + y), all_actions)


    def update_weights(self, weights, state_val, next_move, reward, lr):
        delta = (reward + next_move[1]) - state_val
        for i, _ in enumerate(weights):
            weights[i] = weights[i] + (lr * delta * next_move[2][i])
        return weights

    # features (in order) [mdist to mon, mdist to bomb, mdist to exit]
    def get_features(self, wrld, loc):
        mdist = self.monster_dist(wrld, loc)
        edist = self.exit_dist(wrld, loc)
        bdist = self.bomb_dist(wrld, loc)
        features = np.array([mdist, bdist, edist])
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
        return abs(loc[0] - wrld.width()-1) + abs(loc[1] - wrld.height()-1)
