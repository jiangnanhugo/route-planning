import copy
import numpy as np
from collections import defaultdict
import torch

class CRISP_TSP_MASK(object):
    def __init__(self, mdd):
        self.mdd = mdd

        self.n_states = len(self.mdd.mdd)
        self.n_locs = self.mdd.n_locations

        self.max_duration = self.mdd.max_duration
        self.distance_matrix = np.array(self.mdd.distance_matrix)
        self.max_stops = self.mdd.max_stops
        self.startp = self.mdd.startp
        self.endp = self.mdd.endp


        ### state_next, state_mask
        self.state_next = np.zeros((self.n_states, self.n_locs), dtype=np.int)
        self.state_mask = np.zeros((self.n_states, self.n_locs), dtype=np.float)

        for i, node in enumerate(self.mdd.mdd):
            for a, n in zip(node.a, node.n):
                self.state_next[i, a] = n
                self.state_mask[i, a] = 1.0

        ### state latest time
        self.state_latest_time = np.zeros((self.n_states, self.n_locs), dtype=np.float)
        for i, node in enumerate(self.mdd.mdd):
            for a, time in zip(node.a, node.latest_time):
                self.state_latest_time[i, a] = time

        ### state some up
        self.state_some_up = np.zeros((self.n_states, self.n_locs), dtype=np.int)
        for i, node in enumerate(self.mdd.mdd):
            for loc in node.some_up:
                self.state_some_up[i, loc] = 1

    def generate_mask_with_ground_truth(self, real_paths, visit):
        max_stop_seq_out, n_batch = real_paths.shape
        assert max_stop_seq_out == self.max_stops + 1

        # state
        state = np.zeros((n_batch, self.max_stops + 2), dtype=np.int32)
        state[:, 0] = self.mdd.root

        # loc
        loc = np.zeros((n_batch, self.max_stops + 2), dtype=np.int32)
        loc[:, 0] = self.startp

        # time
        time = np.zeros((n_batch, self.max_stops + 2))
        time[:, 0] = 0.

        # visited
        visited = np.zeros((n_batch, self.n_locs), dtype=np.float32)

        # to visit
        to_visit = copy.deepcopy(visit)

        out_mask_all = np.zeros((self.max_stops + 1, n_batch, self.n_locs), dtype=np.float32)

        for i in range(self.max_stops + 1):
            # compute the mask
            mask = self.state_mask[state[:, i]]

            # visited / to visit filter
            out_mask = np.multiply(np.multiply(1.0 - visited, mask), to_visit).clone()

            for j in range(n_batch):
                for a in range(self.n_locs):
                    if out_mask[j,a] > 0:
                        # max_duration filter
                        if time[j,i] + self.distance_matrix[loc[j, i], a] + self.state_latest_time[state[j, i], a] > self.max_duration:
                            out_mask[j,a] = 0.

                        # some down filter
                        next_state = self.state_next[state[j, i], a]
                        for locid in range(self.n_locs):
                            if locid != a and to_visit[j, locid] > 0 and self.state_some_up[next_state,locid] == 0:
                                out_mask[j, a] = 0.
                                break

            out_mask_all[i, :, :] = out_mask
            maxi = real_paths[i,:]

            ## transforms to the next state
            next_state = self.state_next[state[:, i]]

            state[:, i+1] = np.take(next_state, maxi)
            # print('state='+str( state[:,i+1] ))
            # update to_visit, visited, loc, time
            for j in range(n_batch):
                to_visit[j, maxi[j]] = 0.
                visited[j, maxi[j]] = 1.
                loc[j, i+1] = maxi[j]
                time[j, i+1] = time[j, i] + self.distance_matrix[loc[j, i], maxi[j]]

        return out_mask_all

    def generate_route_with_inference(self, seq_out, visit):

        max_stop_seq_out, n_batch, n_locs = seq_out.shape
        seq_out = seq_out.squeeze()
        assert max_stop_seq_out == self.max_stops + 1
        assert n_locs == self.n_locs

        # state
        state = np.zeros(self.max_stops + 2, dtype=np.int32)
        state[0] = self.mdd.root

        # loc
        loc = np.zeros(self.max_stops + 2, dtype=np.int32)
        loc[0] = self.startp

        # time
        time = np.zeros(self.max_stops + 2)
        time[0] = 0.

        # visited
        visited = np.zeros(self.n_locs, dtype=np.float32)

        # to visit
        to_visit = copy.deepcopy(visit).detach().numpy().flatten()

        generated_routes = []
        for i in range(self.max_stops + 1):
            # compute the mask
            mask = self.state_mask[state[i]]

            # visited / to visit filter
            out_mask = np.multiply(np.multiply(1.0 - visited, mask), to_visit)
            for a in range(self.n_locs):
                if out_mask[a] > 0:
                    # max_duration filter
                    if time[i] + self.distance_matrix[loc[i], a] + self.state_latest_time[state[i], a] > self.max_duration:
                        out_mask[a] = 0.

                    # some down filter
                    next_state = self.state_next[state[i], a]
                    for locid in range(self.n_locs):
                        if locid != a and to_visit[locid] > 0 and self.state_some_up[next_state,locid] == 0:
                            out_mask[a] = 0.
                            break

            current_out = np.multiply(seq_out[i,:], out_mask)
            normalized_vars_predict = current_out / np.sum(current_out, keepdims=True)
            normalized_vars_predict[np.isnan(normalized_vars_predict)] = 0.0

            # print("prob={}".format(seq_out[i,:]))
            # print("normalized={}".format(normalized_vars_predict))
            maxi = np.argmax(normalized_vars_predict, axis=0)
            sampled_loc=random_sample_with_majority_voting(normalized_vars_predict)
            generated_routes.append(sampled_loc)
            # print("maxi {}, sampled maxi: {}".format(maxi, sampled_loc))
            # print("maxi={}".format(maxi))

            ## transforms to the next state
            next_state = self.state_next[state[i]]

            state[i+1] = np.take(next_state, sampled_loc)
            # print('state='+str( state[:,i+1] ))
            # update to_visit, visited, loc, time
            to_visit[sampled_loc] = 0.
            visited[sampled_loc] = 1.
            loc[i+1] = sampled_loc
            time[i+1] = time[i] + self.distance_matrix[loc[i], sampled_loc]

        return generated_routes


def convert_prob_to_range(probs):
    sumed=0.
    histgram=[0., ]
    for p in probs:
        sumed+=p
        histgram.append(sumed)
    ranges=[]
    for i in range(len(histgram)-1):
        ranges.append((histgram[i], histgram[i+1]))
    return ranges


def get_location_from_prob_range(ranges, z):
    for i, (left, right) in enumerate(ranges):
        if left == right:
            continue
        if z >= left and z < right:
            return i
    return 0


def random_sample_with_majority_voting(probs, num_of_tryouts=100):
    probs=probs.flatten()
    ranges=convert_prob_to_range(probs)
    random_zs=np.random.random(num_of_tryouts)
    majority_vote=defaultdict(int)
    for z in random_zs:
        picked_loc = get_location_from_prob_range(ranges, z)
        majority_vote[picked_loc] += 1

    sort_majority_vote= sorted(majority_vote.items(), key=lambda x: x[1], reverse=True)
    return sort_majority_vote[0][0]