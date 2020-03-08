# This is necessary to find the main code
import collections
import heapq
import math
import random
import sys

# from Bomberman.bomberman.entity import CharacterEntity
# from Bomberman.bomberman.events import Event
# from Bomberman.bomberman.world import World
from astarsearch import a_star_search, bfs

sys.path.insert(0, '../bomberman')
# Import necessary stuff
from events import Event
from entity import CharacterEntity
from world import World
from colorama import Fore, Back

from itertools import product, starmap
from collections import Counter

Action = collections.namedtuple('Action', 'dx dy bomb')


class ApproxQCharacter(CharacterEntity):

    def __init__(self, name, avatar, x, y):
        super().__init__(name, avatar, x, y)

        self.alpha = 0.1  # Learning Rate
        self.epsilon = 0.05  # Exploration Rate
        self.gamma = 1.0  # Discount Factor

        self.weights = Counter()

    def do(self, state: World):
        next_state, events = state.next()
        monster = bfs(state, (self.x, self.y), next_state.monsters_at)
        explosion = bfs(state, (self.x, self.y), next_state.explosion_at)
        if (monster is None or monster >= 3) and (explosion is None or explosion >= 2):
            path = a_star_search(state, (self.x, self.y), (state.width() - 1, state.height() - 1))
            if not path:
                self.place_bomb()
            else:
                dx = path[1][0] - self.x
                dy = path[1][1] - self.y
                self.move(dx, dy)
        else:
            action = self.get_action(state)

            self.move(action.dx, action.dy)
            if action.bomb:
                self.place_bomb()

            next_state, events = state.next()
            self.update(state, action, next_state, events)

    def next_position(self, action: Action) -> (int, int):
        return self.x + action.dx, self.y + action.dy

    def get_features(self, state: World, action: Action) -> Counter:
        """
        Get the features of the provided state.
        """
        features = Counter()

        next_position = self.next_position(action)

        # Exit
        exit_distance = bfs(state, next_position, state.exit_at)
        if exit_distance:
            features["exit-distance"] = 1 / exit_distance

        # # Bomb
        # bomb_distance = bfs(state, next_position, state.bomb_at)
        # if bomb_distance:
        #     features["bomb-distance"] = 1 / bomb_distance

        # Explosion
        explosion_distance = bfs(state, next_position, state.explosion_at)
        if explosion_distance:
            features["explosion-distance"] = 1 / explosion_distance

        # Monster
        monster_distance = bfs(state, next_position, state.monsters_at)
        if monster_distance:
            features["monster-distance"] = 1 / monster_distance

        return features

    def v(self, state: World) -> float:
        """
        This is the max Q(s,a) of all possible actions
        """
        max_value = -math.inf
        for action in self.get_possible_actions(state):
            max_value = max(max_value, self.q(state, action))
        return max_value

    def q(self, state: World, action: Action) -> float:
        """
        This is the Q or quality function. It describes the quality of being in a particular state (s) and taking a
        particular action (a). The higher the quality, the better the state/action pair.

        Q(s,a) = w1f1(s,a) + w2f2(s,a) + ... + wnfn(s,a)
        """
        features = self.get_features(state, action)
        value = 0
        for feature in features:
            value += self.weights[feature] * features[feature]
        return value

    def update(self, state: World, action: Action, next_state: World, events):
        reward = self.get_reward(state, action, next_state, events)
        features = self.get_features(state, action)
        print(f"Updating with reward: {reward}; features: {features}")

        delta = (reward + self.gamma * self.v(next_state)) - self.q(state, action)
        for feature in features:
            self.weights[feature] += self.alpha * delta * features[feature]

    @staticmethod
    def get_all_actions():
        return map(lambda t: Action(*t), product(tuple(range(-1, 1 + 1)), tuple(range(-1, 1 + 1)), (True, False)))

    @staticmethod
    def is_legal_location(state: World, location: (int, int)):
        return 0 <= location[0] < state.width() and 0 <= location[1] < state.height()

    def get_possible_actions(self, state: World):
        """
        Get the list of possible actions with the given world state.

        An action is a dx, dy, and place bomb tuple.
        """
        return filter(lambda action: self.is_legal_location(state, (action.dx + self.x, action.dy + self.y))
                                     and not state.wall_at(self.x + action.dx, self.y + action.dy),
                      self.get_all_actions())

    def get_action_random(self, state: World):
        actions = self.get_possible_actions(state)
        return random.choice(list(actions))

    def get_action_from_q_values(self, state: World):
        actions = self.get_possible_actions(state)
        if actions:
            max_value = -math.inf
            best_action = None
            for action in actions:
                q = self.q(state, action)
                if q > max_value:
                    max_value = q
                    best_action = action
            return best_action
        return None

    def get_action(self, state: World) -> Action:
        if random.random() > self.epsilon:
            return self.get_action_from_q_values(state)
        return self.get_action_random(state)

    def get_reward(self, state: World, action: Action, next_state: World, events) -> float:
        reward = 0.0

        if action.bomb:
            reward -= 0.1

        for event in events:
            if event.tpe == Event.BOMB_HIT_CHARACTER:
                reward -= 100000
            if event.tpe == Event.CHARACTER_KILLED_BY_MONSTER:
                reward -= 100000
            if event.tpe == Event.CHARACTER_FOUND_EXIT:
                reward += 100

        return reward
