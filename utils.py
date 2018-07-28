from game import wrapped_flappy_bird
import numpy as np
import cv2
import config
import random
import tensorflow as tf

def process_game_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resize = cv2.resize(gray,config.INPUT_SIZE)
    norm_resize = resize/127.
    return norm_resize


def get_next_state(game_state, action, state):
    game_frame, reward, terminal = game_state.frame_step(action)
    norm_resized = process_game_frame(game_frame)
    next_state = np.append(norm_resized.reshape([1,84,84,1]),state[:,:,:,:3], axis=3)
    return next_state, reward, terminal

def get_transition(game_state,action,state):
    next_state, reward, terminal = get_next_state(game_state, action, state)
    return [state, action, reward, next_state, terminal]

def get_random_action():
    action = np.zeros([2],dtype=np.int32)
    index = random.randint(0,1)
    action[index] = 1
    return action

def get_predict_action(x,prediction,sess,state):
    # x = tf.placeholder(dtype=tf.float32,shape=(None,84,84,4))
    action = np.zeros([2], dtype=np.int32)
    pred = sess.run(prediction,feed_dict={x:state})
    index = np.argmax(pred)
    action[index] = 1
    return action


def init_replay(game_state,x,sess,prediction,size=5000):
    action = get_random_action()
    init_frame, _, _ = game_state.frame_step(action)
    init_frame = process_game_frame(init_frame)
    init_state = np.stack([init_frame,init_frame,init_frame,init_frame],axis=2).reshape([1,84,84,4])
    curr_state = init_state
    replay = []
    for i in range(size):
        action = get_predict_action(x,prediction,sess,curr_state)
        print(action)
        transition = get_transition(game_state, action, curr_state)
        replay.append(transition)
        curr_state = transition[3]



