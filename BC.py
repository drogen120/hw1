#!/usr/bin/env python

"""
behavior cloning
"""

import pickle
import tensorflow as tf
import numpy as np
import tf_util
import load_policy
import _pickle as pickle
    
tf.logging.set_verbosity(tf.logging.ERROR)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument("data", type=str)
    parser.add_argument("--model", type=str, default='None')
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--output", type=str, default='None')
    parser.add_argument("--percent", type=int, default=100)

    args = parser.parse_args()

    import gym
    env = gym.make(args.envname)
    
    import tflearn

    print('loading data')
    with open(args.data,'rb') as f:
        obs, act = pickle.load(f)
    print('  -> done!')

    if (args.percent < 100):
        n = int(args.percent / 100.0 * obs.shape[0])
        obs = obs[:n, :]
        act = act[:n, :]

    batch_size = 256

    
    #### Build Model
    X = obs
    Y = act.squeeze()
    input_size = X.shape[1]
    hid1 = 32
    hid2 = 16
    output_size = Y.shape[1]
    
    input_layer = tflearn.input_data(shape=[None, input_size])
    layer1 = tflearn.fully_connected(input_layer, hid1, activation='relu',
                                 regularizer='L2', weight_decay=0.001)
    layer2 = tflearn.fully_connected(layer1, hid2, activation='relu',
                                 regularizer='L2', weight_decay=0.001)
    output_layer = tflearn.fully_connected(layer2, output_size, activation='linear')
    
    adam = tflearn.Adam(learning_rate=0.001)
    net = tflearn.regression(output_layer, optimizer=adam, loss='mean_square')
    
    model = tflearn.DNN(net)
    
    if args.model is 'None':
        model.fit(X, Y, batch_size=256, n_epoch=args.epochs, shuffle=True, show_metric=True, run_id='behavior clone: '+args.envname)
        model.save('model/'+args.envname+'.tfl')
    else:
        model.load(args.model)
    #return 
    # evaluate
    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    eval_num = 50
    max_steps = 500
    
    targets = []
    
    with tf.Session():
        tf_util.initialize()

        returns = []
        for i in range(eval_num):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = model.predict(obs[None,:])
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)
            # simulate target performance
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = policy_fn(obs[None,:])
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            targets.append(totalr)
  
    filename = args.envname + '_result.pkl' if args.output is 'None' else args.output   
    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))
    
    with open(filename, 'wb') as f:
        pickle.dump([returns,targets], f)
        

if __name__ == '__main__':
    main()
