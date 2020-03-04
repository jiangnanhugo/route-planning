#!/usr/bin/python

import matplotlib.pyplot as plt
import matplotlib

import numpy as np


def grep_batch(filename):
    b = []
    inp = file(filename, 'r')
    for l in inp.readlines():
        ll = l.strip()
        if ll.startswith('[after ') and ll.endswith('iterations]'): #ll.endswith('mini-batches'):
            tt = ll.split()
            b.append(int(tt[1]))
    inp.close()
    return b

def grep_pct_valid(filename):
    b = []
    inp = file(filename, 'r')
    for l in inp.readlines():
        ll = l.strip()
        if ll.startswith('pct_valid(gen)='):
            tt = ll.split()
            b.append(float(tt[1]))
    inp.close()
    return b

def grep_pct_loc_invalid(filename):
    b = []
    inp = file(filename, 'r')
    for l in inp.readlines():
        ll = l.strip()
        if ll.startswith('pct_loc_invalid(gen)='):
            tt = ll.split()
            b.append(float(tt[1]))
    inp.close()
    return b

def grep_pct_dis_invalid(filename):
    b = []
    inp = file(filename, 'r')
    for l in inp.readlines():
        ll = l.strip()
        if ll.startswith('pct_dis_invalid(gen)='):
            tt = ll.split()
            b.append(float(tt[1]))
    inp.close()
    return b

def grep_pct_strict_subset(filename):
    b = []
    inp = file(filename, 'r')
    for l in inp.readlines():
        ll = l.strip()
        if ll.startswith('pct_strict_subset(gen)='):
            tt = ll.split()
            b.append(float(tt[1]))
    inp.close()
    return b

def grep_pct_empty(filename):
    b = []
    inp = file(filename, 'r')
    for l in inp.readlines():
        ll = l.strip()
        if ll.startswith('pct_empty(gen)='):
            tt = ll.split()
            b.append(float(tt[1]))
    inp.close()
    return b

def grep_reward_gen(filename):
    b = []
    inp = file(filename, 'r')
    for l in inp.readlines():
        ll = l.strip()
        if ll.startswith('average_reward(gen)='):
            tt = ll.split()
            b.append(float(tt[1]))
    inp.close()
    return b

def grep_reward_real(filename):
    b = []
    inp = file(filename, 'r')
    for l in inp.readlines():
        ll = l.strip()
        if ll.startswith('average_reward(real)='):
            tt = ll.split()
            b.append(float(tt[1]))
    inp.close()
    return b

def grep_kl_gen(filename):
    b = []
    inp = file(filename, 'r')
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

# file_mask = "output/bays29_s0_e0_md1000_ms6_random_reward_ne100000_w1000_mask1.out"
# file_no_mask = "output/bays29_s0_e0_md1000_ms6_random_reward_ne100000_w1000_mask0.out"

# file_mask = "output/bays29_s0_e0_md1000_ms6_random_reward_ne100000_w1000.data.mask1.out"
# file_no_mask = "output/bays29_s0_e0_md1000_ms6_random_reward_ne100000_w1000.data.mask0.out"

# output_reward_file = "output/bays29_ms6_exp3_reward"
# output_pct_valid_file = "output/bays29_ms6_exp3_pct_valid"

file_mask = "output/bays29_s0_e0_md1000_ms20_random_reward_ne10000_w1000_gt1000.data.mask1.out"
file_no_mask = "output/bays29_s0_e0_md1000_ms20_random_reward_ne10000_w1000_gt1000.data.mask0.out"

#file_mask = "output/bays29_s0_e0_md1000_ms12_random_reward_ne10000_w1000_gt1000.data.mask1.out"
#file_no_mask = "output/bays29_s0_e0_md1000_ms12_random_reward_ne10000_w1000_gt1000.data.mask0.out"
#output_reward_file = "output/bays29_ms12_exp3_reward_new"
#output_pct_valid_file = "output/bays29_ms12_exp3_pct_valid_new"

#file_mask = "output/bays29_s0_e0_md1000_ms6_random_reward_ne100000_w1000.data.mask1.new.out"
#file_no_mask = "output/bays29_s0_e0_md1000_ms6_random_reward_ne100000_w1000.data.mask0.new.out"

#output_reward_file = "output/bays29_ms6_exp3_reward_new"
#output_pct_valid_file = "output/bays29_ms6_exp3_pct_valid_new"

# plot 1: reward plot
filename = file_no_mask

nbatches0 = grep_batch(filename)
print 'nbatches0=', nbatches0
reward_gen0 = grep_reward_gen(filename)
print 'reward_gen0=', reward_gen0
reward_real0 = grep_reward_real(filename)
print 'reward_real0=', reward_real0

for i in xrange(len(reward_gen0)):
    reward_gen0[i] /=  reward_real0[i]

filename = file_mask

nbatches1 = grep_batch(filename)
print 'nbatches1=', nbatches1
reward_gen1 = grep_reward_gen(filename)
print 'reward_gen1=', reward_gen1
reward_real1 = grep_reward_real(filename)
print 'reward_real1=', reward_real1

for i in xrange(len(reward_gen1)):
    reward_gen1[i] /=  reward_real1[i]

font = {'size'   : 25}
matplotlib.rc('font', **font)

plt.figure()
plt.plot(nbatches0, reward_gen0, 'b-.', label='without WalkDD')
plt.plot(nbatches1, reward_gen1, 'r-*', label='with WalkDD')


plt.xlabel('# mini-batches')
plt.ylabel('Avg Normalized Reward')
plt.ylim((0, 1.5))
plt.legend(loc='lower right')


#plt.show()
plt.savefig(output_reward_file+'.pdf', bbox_inches='tight', pad_inches=0.1)
plt.savefig(output_reward_file+'.jpg', bbox_inches='tight', pad_inches=0.1)


# plot 2: subset, etc, for with mask

filename = file_mask

pct_valid1 = grep_pct_valid(filename)
print 'pct_valid1=', pct_valid1

pct_loc_invalid1 = grep_pct_loc_invalid(filename)
print 'pct_loc_invalid1=', pct_loc_invalid1

pct_dis_invalid1 = grep_pct_dis_invalid(filename)
print 'pct_dis_invalid1=', pct_dis_invalid1

pct_strict_subset1 = grep_pct_strict_subset(filename)
print 'pct_strict_subset1=', pct_strict_subset1

pct_empty1 = grep_pct_empty(filename)
print 'pct_empty1=', pct_empty1

pct_valid1 = np.array(pct_valid1)
pct_loc_invalid1 = np.array(pct_loc_invalid1)
pct_dis_invalid1 = np.array(pct_dis_invalid1)
pct_strict_subset1 = np.array(pct_strict_subset1)
pct_empty1 = np.array(pct_empty1)


plt.figure()
plt.plot(nbatches1, pct_valid1*100., 'r-*', label='Valid Schedules')
plt.plot(nbatches1, pct_loc_invalid1*100., 'g--', label='Permutation Invalid')
plt.plot(nbatches1, pct_dis_invalid1*100., 'b-.', label='Distance Invalid')
plt.plot(nbatches1, pct_strict_subset1*100., 'm-o', label="Valid Nonempty Subset")
plt.plot(nbatches1, pct_empty1*100., 'k-', label='Empty Schedules')

plt.xlabel('# mini-batches')
plt.ylabel('Percentage')
plt.ylim((-5, 101))
plt.title('with WalkDD')
#plt.legend(loc='best', bbox_to_anchor=(0., 1.2, 1., 0.5))

plt.savefig(output_pct_valid_file+'_mask1.pdf', bbox_inches='tight', pad_inches=0.1)
plt.savefig(output_pct_valid_file+'_mask1.jpg', bbox_inches='tight', pad_inches=0.1)


# plot 3: subset, etc, for with no mask

filename = file_no_mask

pct_valid0 = grep_pct_valid(filename)
print 'pct_valid0=', pct_valid0

pct_loc_invalid0 = grep_pct_loc_invalid(filename)
print 'pct_loc_invalid0=', pct_loc_invalid0

pct_dis_invalid0 = grep_pct_dis_invalid(filename)
print 'pct_dis_invalid0=', pct_dis_invalid0

pct_strict_subset0 = grep_pct_strict_subset(filename)
print 'pct_strict_subset0=', pct_strict_subset0

pct_empty0 = grep_pct_empty(filename)
print 'pct_empty0=', pct_empty0

pct_valid0 = np.array(pct_valid0)
pct_loc_invalid0 = np.array(pct_loc_invalid0)
pct_dis_invalid0 = np.array(pct_dis_invalid0)
pct_strict_subset0 = np.array(pct_strict_subset0)
pct_empty0 = np.array(pct_empty0)


plt.figure()
plt.plot(nbatches0, pct_valid0*100., 'r-*', label='Valid Schedules')
plt.plot(nbatches0, pct_loc_invalid0*100., 'g--', label='Permutation Invalid')
plt.plot(nbatches0, pct_dis_invalid0*100., 'b-.', label='Distance Invalid')
plt.plot(nbatches0, pct_strict_subset0*100., 'm-o', label="Valid Nonempty Subset")
plt.plot(nbatches0, pct_empty0*100., 'k-', label='Empty Schedules')

plt.xlabel('# mini-batches')
plt.ylabel('Percentage')
plt.ylim((-5, 101))
plt.title('without WalkDD')

plt.legend(loc='best', bbox_to_anchor=(0., -0.75, 1., 0.5))

plt.savefig(output_pct_valid_file+'_mask0.pdf', bbox_inches='tight', pad_inches=0.1)
plt.savefig(output_pct_valid_file+'_mask0.jpg', bbox_inches='tight', pad_inches=0.1)



# pct_valid0 = grep_pct_valid(filename)
# print 'pct_valid0=', pct_valid0
# pct_valid0 = np.array(pct_valid0)






#plt.plot(nbatches0, pct_valid0*100., 'b-.', label='w/o MDD')


# #plt.show()
# plt.savefig(output_pct_valid_file+'.jpg', bbox_inches='tight', pad_inches=0.1)




