#!/usr/bin/env python

"""
generate figure for task2 question3
"""

import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
import _pickle as pickle
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)

    args = parser.parse_args()
    
    print('loading data ...')
    percents = [20,40,60,80,100]
    dat = []
    tar = []
    for p in percents:
        print('--> percent = %d' % p)
        filename = 'log/{a}_per{b}_res.pkl'.format(a=args.envname,b=p)
        with open(filename,'rb') as f:
            returns,targets = pickle.load(f)
        dat.append(returns)
        tar.append(targets)
    print('  >> done!')
    dat = [dat,tar]
    X = np.transpose(np.array(dat),[2,1,0]) # [batch, len, channels]
    label = [p * 10 for p in percents]
    sns.tsplot(data=X,time=label,condition=['learner','expert'])
    plt.ylabel('returns')
    plt.xlabel('rollouts from experts')
    #plt.title('Performance for {a} with different rollouts from experts.\nrun bash run_all to get the figures.'.format(a=args.envname))
    #plt.show()
    plt.savefig('figures/fig1-{a}.png'.format(a=args.envname))


if __name__ == '__main__':
    main()
