from game import wrapped_flappy_bird
import numpy as np
import cv2
import config
import random
import tensorflow as tf
import pickle
import os
import logging

def get_logger(filepath,level=logging.INFO):
    dir = os.path.dirname(filepath)
    if not os.path.exists(dir):
        os.mkdir(dir)
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    # create a file handler
    handler = logging.FileHandler(filepath)
    handler.setLevel(logging.INFO)

    # create a logging format
    #formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    #handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)
    return logger

#初始化sess,或回复保存的sess
def start_or_restore_training(sess,saver,checkpoint_dir):
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print('Restore the model from checkpoint %s' % ckpt.model_checkpoint_path)
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
        step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
    else:
        sess.run(tf.global_variables_initializer())
        step = 1
        print('start training from new state')
    return sess,step

#预处理游戏帧(灰度化,调整大小,归一化)
def process_game_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resize = cv2.resize(gray,config.INPUT_SIZE)
    norm_resize = resize/127.
    return norm_resize

def save_to_pickle(obj,filepath):
    with open(filepath,'wb') as file:
        pickle.dump(obj,file)

def load_from_pickle(filepath):
    with open(filepath,'rb') as file:
        data = pickle.load(file)
    return data

#根据acton获取下一个state,当前即时rewrad,游戏是否终结terminal
def get_next_state(game_state, action, state):
    game_frame, reward, terminal = game_state.frame_step(action)
    norm_resized = process_game_frame(game_frame)
    next_state = np.append(norm_resized.reshape([1,84,84,1]),state[:,:,:,:3], axis=3)
    return next_state, reward, terminal

#把一个transition的data封装在一个list里
def get_transition(game_state,action,state):
    next_state, reward, terminal = get_next_state(game_state, action, state)
    return [state, action, reward, next_state, terminal]

#随机选择action
def get_random_action():
    action = np.zeros([2],dtype=np.int32)
    index = random.randint(0,1)
    action[index] = 1
    return action

#根据模型预测选择预测reward最大的action
def get_predict_action(x,prediction,sess,state):
    action = np.zeros([2], dtype=np.int32)
    pred = sess.run(prediction,feed_dict={x:state})
    index = np.argmax(pred)
    action[index] = 1
    return action

#epslion 为采取随机action的概率
def get_action(epslion, x, prediction, sess, curr_state):
    if random.random()<epslion:
        action = get_random_action()
    else:
        action = get_predict_action(x, prediction, sess, curr_state)

    return action

#观察环境并收集replay memory
def observation(game_state,esplion,x,sess,prediction,step=config.OBSERVATION_STEP):
    action = get_random_action()
    init_frame, _, _ = game_state.frame_step(action)
    init_frame = process_game_frame(init_frame)
    init_state = np.stack([init_frame,init_frame,init_frame,init_frame],axis=2).reshape([1,84,84,4])
    curr_state = init_state
    replay = []
    for i in range(step):
        action = get_action(esplion,x,prediction,sess,curr_state)
        transition = get_transition(game_state, action, curr_state)
        replay.append(transition)
        curr_state = transition[3]#更新当前state
    return curr_state, replay

#更新replay memory
def update_replay(game_state,replay,esplion,x,prediction,sess,curr_state):
    action = get_action(esplion, x, prediction, sess, curr_state)
    transition = get_transition(game_state, action, curr_state)
    if config.REPLAY_MEMORY<len(replay):
        replay.pop(0)
    replay.append(transition)
    next_state = transition[3]
    return replay, next_state

#初始化sess,或回复保存的sess
def start_or_restore_training(sess,saver,checkpoint_dir):
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print('Restore the model from checkpoint %s' % ckpt.model_checkpoint_path)
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
        step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
    else:
        sess.run(tf.global_variables_initializer())
        step = 1
        print('start training from new state')
    return sess,step

#初始化，或回复game_state,replay,curr_state,esplion等
def init_or_restore_training_obj(savefile,x, sess, prediction):
    if os.path.exists(savefile):
        save_obj = load_from_pickle(savefile)
        game_state = save_obj[0]
        replay = save_obj[1]
        curr_state = save_obj[2]
        esplion = save_obj[3]
    else:
        esplion = config.ESPLION
        game_state = wrapped_flappy_bird.GameState()
        curr_state, replay = observation(game_state, esplion, x, sess, prediction, step=config.OBSERVATION_STEP)

    return game_state,replay,curr_state,esplion
