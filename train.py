import tensorflow as tf
import utils
import network
import config
from game import wrapped_flappy_bird
import random
import numpy as np
import time

x = tf.placeholder(dtype=tf.float32,shape=(None,84,84,4),name='input_x')
y = tf.placeholder(dtype=tf.float32,shape=(None),name='input_y')
a = tf.placeholder(dtype=tf.float32,shape=(None,2),name='input_a')
prediction = network.network(x)
pred_reward = tf.reduce_sum(tf.multiply(prediction,a))
loss = tf.reduce_mean(tf.square(pred_reward - y),name='loss')
train_step = tf.train.AdamOptimizer().minimize(loss)
saver = tf.train.Saver()
logger = utils.get_logger(config.LOGFILE)
with tf.Session() as sess:
    # 初始化sess,或回复保存的sess
    sess,step = utils.start_or_restore_training(sess,saver,checkpoint_dir=config.CHECKDIR)

    game_state, replay, curr_state, esplion = utils.init_or_restore_training_obj(config.SAVEFILE,x,sess,prediction)


    start_time = time.time()
    while(True):

        #在replay memory 中随机抽取batch,进行训练
        batch = random.sample(replay,config.BATCH_SIZE)

        state_batch = np.array([tran[0] for tran in batch]).reshape([-1, 84, 84, 4])
        action_batch = np.array([tran[1] for tran in batch])
        curr_reward_batch = np.array([tran[2] for tran in batch])
        next_state_batch = np.array([tran[3] for tran in batch]).reshape([-1, 84, 84, 4])
        terminal_batch = np.array([tran[4] for tran in batch])

        reward_futher_batch = sess.run(prediction, feed_dict={x: next_state_batch})
        max_further_reward_batch = np.max(reward_futher_batch, axis=1)
        target_reward_batch = curr_reward_batch + (-terminal_batch * max_further_reward_batch)

        sess.run(train_step,feed_dict={x:state_batch, y:target_reward_batch, a:action_batch})

        # 更新replay memory
        replay,next_state = utils.update_replay(game_state,replay,esplion,x,prediction,sess,curr_state)
        #更新curr_state
        curr_state = next_state
        #更新step
        step+=1

        #esplion随着训练次数衰减,使采取随机action的概率越来越低
        if esplion>config.FINIAL_ESPLION:
            esplion -= (config.ESPLION - config.FINIAL_ESPLION)/config.EXPLORE

        if step%1000==0:
            train_loss = sess.run(loss,feed_dict={x:state_batch, y:target_reward_batch, a:action_batch})
            duration = time.time() - start_time
            logger.info("step %d: loss is %g (%0.3f sec)" % (step, train_loss, duration))
            start_time = time.time()
        if step%10000==0:
            saver.save(sess, config.CHECKFILE, global_step=step)
            utils.save_to_pickle([game_state,replay,curr_state,esplion])
            print('writing checkpoint at step %s' % step)








