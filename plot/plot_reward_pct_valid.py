#!/usr/bin/python

import matplotlib.pyplot as plt
import matplotlib

import numpy as np


def grep_batch(filename):
    b = []
    inp = open(filename, 'r')
    for l in inp.readlines():
        ll = l.strip()
        if ll.startswith('[after ') and ll.endswith('iterations]'): #ll.endswith('mini-batches'):
            tt = ll.split()
            b.append(int(tt[1]))
    inp.close()
    return b

def grep_pct_valid(filename):
    b = []
    inp = open(filename, 'r')
    for l in inp.readlines():
        ll = l.strip()
        if ll.startswith('pct_valid(gen)='):
            tt = ll.split()
            b.append(float(tt[1]))
    inp.close()
    return b

def grep_reward_gen(filename):
    b = []
    inp = open(filename, 'r')
    for l in inp.readlines():
        ll = l.strip()
        if ll.startswith('average_reward(gen)='):
            tt = ll.split()
            b.append(float(tt[1]))
    inp.close()
    return b

def grep_reward_real(filename):
    b = []
    inp = open(filename, 'r')
    for l in inp.readlines():
        ll = l.strip()
        if ll.startswith('average_reward(real)='):
            tt = ll.split()
            b.append(float(tt[1]))
    inp.close()
    return b

def grep_kl_gen(filename):
    b = []
    inp = open(filename, 'r')
    for l in inp.readlines():
        ll = l.strip()
        if ll.startswith('kl(gen)='):
            tt = ll.split()
            b.append(float(tt[1]))
    inp.close()
    return b


##### Main Program

# file_mask = "output/6city_exp1_mask1.out"
# file_no_mask = "output/6city_exp1_mask0.out"

#file_mask = "output/6city_exp1.data.mask1.out"
#file_no_mask = "output/6city_exp1.data.mask0.out"

file_mask = "output/bays29_s0_e0_md1000_ms20_random_reward_ne10000_w1000_gt1000.data.mask1.out"
file_no_mask ="output/bays29_s0_e0_md1000_ms20_random_reward_ne10000_w1000_gt1000.data.mask0.out"

#output_pct_valid_file = "output/6city_exp1_pct_valid_new"
#output_reward_file = "output/6city_exp1_reward_new"

output_pct_valid_file = "output/bays29_exp1_pct_valid"
output_reward_file = "output/bays29_exp1_reward"

# plot 1
filename = file_no_mask

nbatches0 = grep_batch(filename)
print('nbatches0=', nbatches0)
pct_valid0 = grep_pct_valid(filename)
print('pct_valid0=', pct_valid0)
reward_gen0 = grep_reward_gen(filename)
print('reward_gen0=', reward_gen0)
reward_real0 = grep_reward_real(filename)
print('reward_real0=', reward_real0)

for i in range(len(reward_gen0)):
    reward_gen0[i] /=  reward_real0[i]

pct_valid0 = np.array(pct_valid0)


filename = file_mask

nbatches1 = grep_batch(filename)
print('nbatches1=', nbatches1)
pct_valid1 = grep_pct_valid(filename)
print('pct_valid1=', pct_valid1)
reward_gen1 = grep_reward_gen(filename)
print('reward_gen1=', reward_gen1)
reward_real1 = grep_reward_real(filename)
print('reward_real1=', reward_real1)

for i in range(len(reward_gen1)):
    reward_gen1[i] /=  reward_real1[i]

pct_valid1 = np.array(pct_valid1)


font = {'size': 20}
matplotlib.rc('font', **font)

plt.figure()
plt.plot(nbatches0, reward_gen0, 'b-.', label='w/o WalkDD')
plt.plot(nbatches1, reward_gen1, 'r-*', label='w WalkDD')

plt.xlabel('# mini-batches')
plt.ylabel('Avg Normalized Reward')
plt.legend()

#plt.show()
plt.savefig(output_reward_file+'.pdf', bbox_inches='tight', pad_inches=0.1)
plt.savefig(output_reward_file+'.jpg', bbox_inches='tight', pad_inches=0.1)

plt.figure()
plt.plot(nbatches0, pct_valid0*100., 'b-.', label='w/o WalkDD')
plt.plot(nbatches1, pct_valid1*100., 'r-*', label='w WalkDD')

plt.xlabel('# mini-batches')
plt.ylabel('% Valid Schedules')
plt.legend(loc=7)

#plt.show()
plt.savefig(output_pct_valid_file+'.pdf', bbox_inches='tight', pad_inches=0.1)
plt.savefig(output_pct_valid_file+'.jpg', bbox_inches='tight', pad_inches=0.1)


