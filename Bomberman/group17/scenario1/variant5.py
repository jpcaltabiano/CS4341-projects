# This is necessary to find the main code
import sys
sys.path.insert(0, '../../bomberman')
sys.path.insert(1, '..')

# Import necessary stuff
import random
from game import Game
from monsters.stupid_monster import StupidMonster
from monsters.selfpreserving_monster import SelfPreservingMonster
from interactivecharacter import InteractiveCharacter

# TODO This is your code!
sys.path.insert(1, '../group17')
from approxqcharacter import ApproxQCharacter

# Create the game
random.seed(123) # TODO Change this if you want different random choices
g = Game.fromfile('map.txt')
g.add_monster(StupidMonster("stupid", # name
                            "S",      # avatar
                            5, 1,     # position
))
g.add_monster(SelfPreservingMonster("aggressive", # name
                                    "A",          # avatar
                                    3, 13,        # position
                                    2             # detection range
))


g.add_character(ApproxQCharacter("me", # name
                              "C",  # avatar
                              3, 4  # position
))

# Run!
g.go()
