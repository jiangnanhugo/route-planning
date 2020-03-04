#!/usr/bin/python

import matplotlib.pyplot as plt
import matplotlib
font = {'size': 18}
matplotlib.rc('font', **font)


import numpy as np

def grep_reward_gen(filename):
    b = []
    inp = open(filename, 'r').read().split("\n")
    for l in inp:
        ll = l.strip()
        tt = ll.split("=")
        if len(tt)>1:
            b.append(float(tt[1]))
    return b

for loc in range(9,30):
    file_mask= "dataset/"+str(loc)+"locations/normalized_reward_mask1.out"
    file_no_mask="dataset/"+str(loc)+"locations/normalized_reward_mask0.out"
    output_reward_file="dataset/"+str(loc)+"reward"
    reward_gen0 = grep_reward_gen(file_no_mask)
    reward_gen1 = grep_reward_gen(file_mask)
    print("push!(noreward,",reward_gen1,end=')\n')

    plt.figure()
    plt.plot(range(len(reward_gen0)), reward_gen0, 'b-.', label='w/o DD')
    plt.plot(range(len(reward_gen1)), reward_gen1, 'r-*', label='w DD')

    plt.xlabel('training epoch')
    plt.ylabel('normalized Reward')
    plt.legend()
    plt.savefig(output_reward_file+'.jpg', bbox_inches='tight', pad_inches=0.1)
    plt.close()
