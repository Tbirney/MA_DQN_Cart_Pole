import sys
sys.path.append('..')

from game_envs.multi_agent_cart_upright import GameInstance
import numpy as np
from nn import deep_q_nn
import pygame

NUM_SENSORS = 8


def play(cart1_model, cart2_model):

    game_state = GameInstance()

    # Do nothing to get initial.
    state = None
    for i in range(10):
        cart1_reward, cart2_reward, state = game_state.frame_step([2, 2])

    # Move.
    while True:
        # if game_state.car.body.position[0] < 0 or\
        #     game_state.car.body.position[0] > cart_pendulum_upright.SCREEN_WIDTH:
        #     pygame.display.quit()
        #     pygame.quit()
        #     game_state = cart_pendulum_upright.GameInstance()
        #     _, state = game_state.frame_step(None)
        #     continue
        
        # angle = state[0][0]
        # if (angle >=0 and angle <= 30) or \
        #    (angle >=330 and angle <= 360):
        #     pygame.display.quit()
        #     pygame.quit()
        #     game_state = cart_pendulum_upright.GameInstance()
        #     for i in range(10):
        #         _, state = game_state.frame_step(None)
        #     continue
        


        # Choose action.
        cart1_action = (np.argmax(cart1_model.predict(state, batch_size=1)))
        cart2_action = (np.argmax(cart2_model.predict(state, batch_size=1)))

        # Take action.
        cart1_reward, cart2_reward, state = game_state.frame_step([cart1_action, cart2_action])


if __name__ == "__main__":
    cart1_saved_model = '../models/multi-agent-upright-multi-dqn-cart1/128-128-64-10000-100000.h5'
    cart2_saved_model = '../models/multi-agent-upright-multi-dqn-cart2/128-128-64-10000-100000.h5'
    cart1_model = deep_q_nn(NUM_SENSORS, [128, 128], cart1_saved_model)
    cart2_model = deep_q_nn(NUM_SENSORS, [128, 128], cart2_saved_model)
    play(cart1_model, cart2_model)