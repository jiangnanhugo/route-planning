#!/usr/bin/python

import random, copy, pickle, argparse
from itertools import product
import numpy as np
import mdd
import data_utils
from filter_refine_mdd import read_tsp_file_full_matrix


def random_sample_loc_set(mdd, root, visited, rewards):
    if root == 0:
        return set([]), 0.
    else:
        node = mdd.mdd[root]
        n_choices = len(node.a)

        choice = random.randint(0, n_choices-1)
        while node.a[choice] in visited:
            choice = random.randint(0, n_choices-1)

        loc_set, cur_reward = random_sample_loc_set(mdd, node.n[choice], (visited | set([node.a[choice]])), rewards)

        return ( loc_set | set([node.a[choice]]) ), cur_reward + rewards[len(visited)][node.a[choice]]


def dfs_best_loc_visitings(mdd, root, visit, idx, to_visit, rewards, cur_reward):
    global best_rewards, best_visits

    if root == 0:
        if cur_reward > best_rewards:
            for i in range(idx):
                best_visits[i] = visit[i]
            best_rewards = cur_reward
    else:
        node = mdd.mdd[root]
        for i, ai in enumerate(node.a):
            ni = node.n[i]
            if (ai in to_visit) and \
               (mdd.mdd[ni].some_up | set([ai])).issuperset(to_visit) and \
               (not ai in visit[:idx]):

                visit[idx] = ai
                dfs_best_loc_visitings(mdd, ni, visit, idx+1, \
                                       to_visit - set([ai]), \
                                       rewards,
                                       cur_reward + rewards[idx][ai])


def dfs_best_loc_visitings_v2(mdd, root, visit, idx, to_visit, \
                              rewards, cur_reward, cur_dist):
    global best_rewards, best_visits, startp, paired_shortest_path, max_duration

    if root == 0:
        # if cur_dist > max_duration:
        #     continue
        #print('here')
        if cur_dist <= max_duration and cur_reward > best_rewards:
            for i in range(idx):
                best_visits[i] = visit[i]
            best_rewards = cur_reward
    else:
        #print('to_visit', to_visit)
        node = mdd.mdd[root]
        for i, ai in enumerate(node.a):
            ni = node.n[i]
            if (ai in to_visit) and \
               (mdd.mdd[ni].some_up | set([ai])).issuperset(to_visit) and \
               (not ai in visit[:idx]):

                if idx == 0:
                    cur_loc = startp
                else:
                    cur_loc = visit[idx-1]

                visit[idx] = ai
                dfs_best_loc_visitings_v2(mdd, ni, visit, idx+1, \
                                          to_visit - set([ai]), \
                                          rewards,\
                                          cur_reward + rewards[idx][ai],\
                                          cur_dist + paired_shortest_path[cur_loc][ai])


def bfs_best_loc_greedy_visitings(mdd, to_visit0, rewards, greedy_trial, paired_dist, max_duration):
    this_layer = [[mdd.mdd[mdd.root], startp, to_visit0, [], 0., 0.]]

    for idx in range(len(to_visit0)):
        next_layer = []
        for r in this_layer:

            node = r[0]
            cur_loc = r[1]
            to_visit = r[2]
            visit = r[3]
            cur_reward = r[4]
            cur_dist = r[5]

            for i, ai in enumerate(node.a):
                ni = node.n[i]
                if (ai in to_visit) and \
                   (mdd.mdd[ni].some_up | set([ai])).issuperset(to_visit) and \
                   (not ai in visit[:idx]) and\
                   (cur_dist + paired_dist[cur_loc][ai] <= max_duration):
                    if idx < len(to_visit0) -1 and ni == 0:
                        continue
                    next_node = [mdd.mdd[ni], ai, to_visit - set([ai]), visit + [ai], cur_reward + rewards[idx][ai], cur_dist + paired_dist[cur_loc][ai]]
                    next_layer.append(next_node)

        if len(next_layer) > greedy_trial:
            next_layer.sort(key=lambda node : node[3], reverse=True)
            next_layer = next_layer[:greedy_trial]

        this_layer = next_layer

    this_layer.sort(key=lambda node : node[4], reverse=True)

    if len(this_layer) == 0:
        return [], -1000.0

    assert this_layer[0][0] == mdd.mdd[0]
    assert len(this_layer[0][2]) == 0
    return this_layer[0][3], this_layer[0][4]


def floyd_warshall(paired_dist):
    path = copy.copy(paired_dist)
    n = len(paired_dist)
    rn = range(n)
    for k, i, j in product(rn, repeat=3):
        path_ik_to_kj = path[i][k] + path[k][j]
        if path[i][j] > path_ik_to_kj:
            path[i][j] = path_ik_to_kj
    return path

##### Main Program #####

# pair_dist
# startp
# endp
# max_duration
# max_stops

# rewards:: (i,j)-th entry: the reward to visit location j in the i-th time slot
# num_examples

# output_file

#### TSPLIB bays29.tsp
tsp_file = "./benchmarks/TSPLIB95/bays29.tsp"
pairwise_dist = read_tsp_file_full_matrix(tsp_file)

parser = argparse.ArgumentParser(description="gen dataset.")
parser.add_argument("--max_stops", required=True, help='maximum stops.')
parser.add_argument("--max_duration", required=True, help='maximum length.')

args = parser.parse_args()
paired_shortest_path = floyd_warshall(pairwise_dist)

startp = 0
endp = 0

max_duration = int(args.max_duration)
max_stops = int(args.max_stops)
max_width = 1000


output_prob_instance = "dataset/"+str(max_stops)+"locations/bays29_s0_e0_md1000_ms"+str(max_stops)+"_random_reward.prob"

num_examples = 10000
greedy_trial = 1000
output_file = "dataset/"+str(max_stops)+"locations/bays29_s0_e0_md1000_ms"+str(max_stops)+"_random_reward_ne10000_w1000_gt1000.data"

# reward
num_locs = len(paired_shortest_path)
print("paired_shortest_path={}".format(num_locs))

def gen_random_rewards(max_stops, num_locs):
    return np.random.rand(max_stops, num_locs)
rewards = gen_random_rewards(max_stops + 1, num_locs)


prob = data_utils.ScheduleProb()
prob.init_by_assign(paired_shortest_path, startp, endp, max_duration, max_stops, rewards)
oup = open(output_prob_instance, 'wb')
pickle.dump(prob, oup)
oup.close()

mdd = mdd.MDD_TSP(paired_shortest_path, startp, endp, max_duration, max_stops, max_width)

mdd.filter_refine_preparation()

# mdd.print_mdd(sys.stdout)
mdd.filter_refine()

print('===== after filter and refining =====')
#mdd.print_mdd(sys.stdout)

#
print("maximum stops", max_stops)
print('===== generated data =====')

oup = open(output_file, 'w')
num_complete = 0
num_rejected = 0
while num_complete < num_examples:
    loc_set, sample_reward = random_sample_loc_set(mdd, mdd.root, set([]), rewards)
    if len(loc_set) > 1:
        best_visits, best_rewards = bfs_best_loc_greedy_visitings(mdd, loc_set, rewards, greedy_trial,
                                                                  paired_shortest_path, max_duration)

        if best_rewards >= 0:
            print(len(best_visits), end=' ', file=oup)
            for loc in best_visits:
                print(loc, end=' ', file=oup)
            print(best_rewards, file=oup)
            # print('sample_reward', sample_reward)

            num_complete += 1
            if num_complete % 100 == 0:
                print('    ', num_complete, 'completed!', '    ', num_rejected, 'rejected.')
        else:
            num_rejected += 1

oup.close()
