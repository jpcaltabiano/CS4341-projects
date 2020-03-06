# This is necessary to find the main code
import sys
sys.path.insert(0, '../../bomberman')
sys.path.insert(1, '..')

# Import necessary stuff
import random
from game import Game
from monsters.stupid_monster import StupidMonster
from monsters.selfpreserving_monster import SelfPreservingMonster

# TODO This is your code!
sys.path.insert(1, '../groupNN')
from approxqcharacter import ApproxQCharacter
import pandas as pd
import numpy as np

# Create the game
random.seed() # TODO Change this if you want different random choices

ws = [0, 0, 0, 0]
epochs = 50
ws_history = np.array((4, 1))
# pd.DatFrame(columns=['w1', 'w2', 'w3', 'w4'])

for i in range(0, epochs):
    g = Game.fromfile('map.txt')
    g.add_monster(StupidMonster("stupid", # name
                                "S",      # avatar
                                3, 5,     # position
    ))
    g.add_monster(SelfPreservingMonster("aggressive", # name
                                        "A",          # avatar
                                        3, 13,        # position
                                        2             # detection range
    ))

    ours = ApproxQCharacter("me", # name
                                "C",  # avatar
                                0, 0  # position
    )
    ours.ws = ws

    g.add_character(ours)

    # Run!
    g.go(1)

    ws = ours.ws
    ws_history = np.append(ws_history, ws)

print("\n\n\nDONE!\n\n\n")
print(ws_history)