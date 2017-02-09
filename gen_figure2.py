#!/usr/bin/env python

"""
generate figure for task3
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
    print('loading expert data ...')
    filename = 'log/{a}_per100_res.pkl'.format(a=args.envname)
    with open(filename,'rb') as f:
        _,targets=pickle.load(f)
    print('  >> done!')
    print('loading dagger data ...')
    filename = args.envname + '-v1_dagger_result.pkl'
    with open(filename,'rb') as f:
        ret_avg,ret_std,ret_all=pickle.load(f)
    print('  >> done!')
    n_dagger = len(ret_avg)
    tar = np.transpose(np.array([targets] * n_dagger),[1,0])
    dat = np.transpose(np.array(ret_all),[1,0])
    print(tar.shape)
    print(dat.shape)
    X = np.transpose(np.array([dat,tar]),[1,2,0]) # [batch, len, channels]
    label = list(range(n_dagger))
    sns.tsplot(data=X,time=label,condition=['learner','expert'])
    plt.ylabel('returns')
    plt.xlabel('DAgger iterations')
    #plt.title('Performance for {a} using DAgger.\nrun bash run_all.sh to get the figures.'.format(a=args.envname))
    #plt.show()
    plt.savefig('figures/fig2-{a}.png'.format(a=args.envname))


if __name__ == '__main__':
    main()
