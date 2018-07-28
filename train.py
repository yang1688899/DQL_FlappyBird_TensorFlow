import tensorflow as tf
import utils
import network
import config
from game import wrapped_flappy_bird

x = tf.placeholder(dtype=tf.float32,shape=(None,84,84,4),name='input')

prediction = network.network(x)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    game_state = wrapped_flappy_bird.GameState()
    replay = utils.init_replay(game_state,x,sess,prediction)
    print(replay.shape)
