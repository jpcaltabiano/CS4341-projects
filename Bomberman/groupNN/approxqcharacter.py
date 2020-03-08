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
from collections import Counter

from sklearn.preprocessing import StandardScaler
import numpy as np
import random
from sensed_world import SensedWorld

class ApproxQCharacter(CharacterEntity):

    def __init__(self, name, avatar, x, y):
        super().__init__(name, avatar, x, y)
        self.alpha = 0.01
        self.epsilon = 0.3

        self.ws = [0, 0, 0, 0, 0, 0, 0, 0]
        self.visited = Counter()
        self.exitSuccess = 0
        self.monsterKilled = 0
        self.wallExploded = 0

    def do(self, wrld):
        state_fts = self.get_features(wrld, (self.x, self.y))
        state_val = (state_fts * self.ws).sum()

        if random.random() > self.epsilon:
            move = self.choose_best_move(wrld)
        else:
            move = self.choose_random_move(wrld)

        # print(move, self.ws)
        print(f"\nI am about to do action: {move[0]}\n")

        if move[0][2] and random.random() > 0.9:
            self.place_bomb()
        self.move(move[0][0], move[0][1])

        print(f"\nI have just done action: {move[0]}\n")
        
        # next_wrld = SensedWorld.from_world(wrld)
        nxt = wrld.next()

        print(f"\nThe next world has been called: {move[0]}\n")

        reward = self.get_reward(nxt[0], nxt[1], move[0], wrld)
        self.ws = self.update_weights(self.ws, state_val, move, reward)
        self.visited[move[0]] += 1


    def get_reward(self, wrld, events, action, old_wrld):
        rw = 0

        if action[1] == -1: rw += -1.0

        # Try to keep things moving
        # TODO: A good idea to keep but is action dx dy, not actual cell coords?
        if action in self.visited:
            rw += -0.1*self.visited[action]
        else:
            rw += 0.04

        for e in events:
            if e.tpe == Event.BOMB_HIT_WALL:
                self.wallExploded += 1
                # rw += 0.05
            if e.tpe == Event.BOMB_HIT_MONSTER:
                self.monsterKilled += 1
                rw += 1.0
            if e.tpe == Event.BOMB_HIT_CHARACTER:
                rw += -1.0
            if e.tpe == Event.CHARACTER_KILLED_BY_MONSTER:
                rw += -0.3
                # slightly smaller to prevent running away from monsters and encourage fwd movement 
            if e.tpe == Event.CHARACTER_FOUND_EXIT:
                print(f"\n\nI have found the exit\n\n")
                print("\nOLD WORLD:")
                old_wrld.printit()
                print("\nNEW WORLD:")
                wrld.printit()
                self.exitSuccess += 1
                rw += 100

        # print("Reward: ", rw)
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


    def update_weights(self, weights, state_val, next_move, reward):
        delta = (reward + next_move[1]) - state_val
        for i, _ in enumerate(weights):
            weights[i] = weights[i] + (self.alpha * delta * next_move[2][i])
        return weights

    # features (in order) [mdist to mon, mdist to bomb, mdist to exit]
    def get_features(self, wrld, loc):
        mdist = self.monster_dist(wrld, loc)
        bdist = self.bomb_dist(wrld, loc)
        threat = self.is_threat(wrld, loc)
        xdist = self.explosion_dist(wrld, loc)
        edist = self.exit_dist(wrld, loc)
        has_bomb = self.has_bomb(wrld, loc)
        has_monster = self.has_monster(wrld, loc)
        has_wall = self.has_wall(wrld, loc)
        features = np.array([mdist, bdist, xdist, threat, 1/edist, has_bomb, has_monster, has_wall])
        # features = np.array([mdist, bdist, xdist, threat, edist])
        normalized = features / np.linalg.norm(features)
        print(normalized)
        return normalized


    def monster_dist(self, wrld, loc):
        # Get the distance to the closest monster
        if not wrld.monsters:
            return 10

        return min(map(lambda monster: abs(loc[0] - monster[0].x) + abs(loc[1] - monster[0].y), wrld.monsters.values()))

    def bomb_dist(self, wrld, loc):
        # Get the distance to the closest bomb
        if not wrld.bombs:
            return 10

        return min(map(lambda bomb: abs(loc[0] - bomb.x) + abs(loc[1] - bomb.y), wrld.bombs.values()))

    def explosion_dist(self, wrld, loc):
        # Get the distance to the closest explosion
        if not wrld.explosions:
            return 10

        return min(map(lambda bomb: abs(loc[0] - bomb.x) + abs(loc[1] - bomb.y), wrld.explosions.values()))

    def is_threat(self, wrld, loc):
        # Is the location threatened
        nxt = wrld
        for i in range(0, wrld.bomb_time):
            nxt = nxt.next()[0]
            if nxt.explosion_at(loc[0], loc[1]):
                return 1

        return 0

    def exit_dist(self, wrld, loc):
        return abs(loc[0] - wrld.width()-1) + abs(loc[1] - wrld.height()-1)

    def has_bomb(self, wrld, loc):
        return 1 if wrld.bomb_at(loc[0], loc[1]) else 0

    def has_monster(self, wrld, loc):
        return 1 if wrld.monsters_at(loc[0], loc[1]) else 0

    def has_wall(self, wrld, loc):
        return wrld.wall_at(loc[0], loc[1])