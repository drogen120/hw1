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
    
max_steps = 500

tf.logging.set_verbosity(tf.logging.ERROR)

def evaluate(env, model):
    print ('Evaluating Current Model ....')
    eval_num = 50
    returns = []
    with tf.Session():
        tf_util.initialize()
        for i in range(eval_num):
            if i % 25 == 0:
                print (' > iter ', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = model.predict(obs[None,:])
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if steps >= max_steps:
                    break
            returns.append(totalr)
    print ("Done!")
    return returns

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument("data", type=str)
    parser.add_argument("--model", type=str, default='None')
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--dagger_step", type=int, default=5)
    parser.add_argument("--dagger_rollout", type=int, default=100)

    args = parser.parse_args()

    import gym
    env = gym.make(args.envname)
    
    import tflearn

    print('loading data')
    with open(args.data,'rb') as f:
        obs, act = pickle.load(f)
    print('  -> done!')

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

    cur = evaluate(env,model)
    ret_avg = [np.mean(cur)]
    ret_std = [np.std(cur)]
    ret_all = [cur]
    for step in range(args.dagger_step):
        print('dagger step #', step)
        observations = []
        actions = []
        with tf.Session():
            tf_util.initialize()

            for i in range(args.dagger_rollout):
                if i % 20 == 0:
                    print ('rollout iter = ', i)
                obs = env.reset()
                done = False
                totalr = 0.
                steps = 0
                while not done:
                    action = model.predict(obs[None,:])
                    obs, r, done, _ = env.step(action)
                    observations.append(obs)
                    actions.append(policy_fn(obs[None,:]))
                    totalr += r
                    steps += 1
                    if steps >= max_steps:
                        break
        
        X = np.row_stack((X, np.array(observations)))
        Y = np.row_stack((Y, np.array(actions).squeeze()))
        model.fit(X, Y, batch_size=256, n_epoch=args.epochs, shuffle=True, show_metric=True, run_id='behavior clone: '+args.envname)
        model.save('model/'+args.envname+'-iter{a}.tfl'.format(a=step))
        returns = evaluate(env,model)
        ret_avg.append(np.mean(returns))
        ret_std.append(np.std(returns))
        ret_all.append(returns)
        print ('current avg return = {a}, std = {b}'.format(a=ret_avg[-1],b=ret_std[-1]))

    
    print('mean return', ret_avg)
    print('std of return', ret_std)
    
    with open(args.envname + '_dagger_result.pkl', 'wb') as f:
        pickle.dump([ret_avg,ret_std,ret_all],f)

if __name__ == '__main__':
    main()
