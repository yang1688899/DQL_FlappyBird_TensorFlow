from game import wrapped_flappy_bird
import numpy as np
import cv2

def get_transition(action):
    game_state = wrapped_flappy_bird.GameState()

    do_nothing = np.zeros(2)
    do_nothing[0] = 1
    game_frame, reward, terminal = game_state.frame_step(do_nothing)



get_transition()