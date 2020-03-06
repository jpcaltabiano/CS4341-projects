# This is necessary to find the main code
import sys
sys.path.insert(0, '../../bomberman')
sys.path.insert(1, '..')

import pygame

# Import necessary stuff
import random
from game import Game
from monsters.stupid_monster import StupidMonster
from monsters.selfpreserving_monster import SelfPreservingMonster

# TODO This is your code!
sys.path.insert(1, '../groupNN')
from approxqcharacter import ApproxQCharacter

# Create the game
random.seed() # TODO Change this if you want different random choices

ws = [0, 0, 0, 0]

for i in range(0, 500):
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
    g.display_gui()
    while not g.done():
        (g.world, g.events) = g.world.next()
        pygame.event.clear()
        g.world.next_decisions()

    ws = ours.ws
    print(f"Game {i}: {ws}")
    
    # g.go(1)


print("\n\n\nDONE!\n\n\n")
print(ws)