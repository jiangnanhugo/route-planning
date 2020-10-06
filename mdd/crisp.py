import copy
import numpy as np
from collections import defaultdict
from mdd.utils import get_mdd


class CRISP(object):
    def __init__(self, n_locations,max_stops, maxwidth):
        self.maxwidth = maxwidth
        self.n_locations = n_locations
        self.max_stops = max_stops
        self.mdd = get_mdd(n_locations, max_stops, maxwidth)

    def generate_mask_with_ground_truth(self, real_paths):
        """
        in training, convert the mdd into mask vector
        :param mdd:
        :param real_paths:  the trajectory
        :param visit: the set of locations in array form. X_i=1 mean the location is in the daily request
        :return:
        """
        max_stop_seq_out, n_batch = real_paths.shape
        # [6, batch_size, 29]
        out_mask_all = np.zeros((self.max_stops + 1, n_batch, self.n_locations), dtype=np.float32)
        for i in range(n_batch):
            filtered_mdd = self.mdd_filtering(self.mdd, real_paths[:,i])
            # print("Input path", real_paths[:,i])
            neighbor_along_path = filtered_mdd.find_neighbor_along_path(real_paths[:,i])
            # print("Output Mask", neighbor_along_path)
            # print(self.idx_to_binary(neighbor_along_path))
            out_mask_all[:,i,:] = self.idx_to_binary(neighbor_along_path)

        return out_mask_all

    # MDD Filtering to Process Daily Requests
    def mdd_filtering(self, MDD, daily_request):
        filtered_mdd = copy.deepcopy(MDD)
        # convert into a set of locations
        daily_request = set(daily_request)
        # print("daily request", daily_request)
        # add the terminal location for type 2 arcs
        daily_request.add(0)
        for j in range(filtered_mdd.numArcLayers):
            nodesinlayer = [v for v in filtered_mdd.allnodes_in_layer(j)]
            for v in nodesinlayer:
                income = [x for x in filtered_mdd.nodes[j][v].incoming]
                for x in income:
                    if x.label not in daily_request:
                        filtered_mdd.remove_arc(x)
                outs = [x for x in filtered_mdd.nodes[j][v].outgoing]
                for x in outs:
                    if x.label not in daily_request:
                        filtered_mdd.remove_arc(x)
        return filtered_mdd

    def idx_to_binary(self, neighbor_along_paths):
        mask = np.zeros(shape=(self.max_stops+1,self.n_locations), dtype=np.float)
        for i,layer in enumerate(neighbor_along_paths):
            for x in layer:
                mask[i,x]=1.0
        return mask


    def generate_route_for_inference(self, seq_out, visit):
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
                    # some down filter
                    next_state = self.state_next[state[i], a]
                    for locid in range(self.n_locs):
                        if locid != a and to_visit[locid] > 0 and self.state_some_up[next_state, locid] == 0:
                            out_mask[a] = 0.
                            break

            current_out = np.multiply(seq_out[i, :], out_mask)
            normalized_vars_predict = current_out / np.sum(current_out, keepdims=True)
            normalized_vars_predict[np.isnan(normalized_vars_predict)] = 0.0

            # print("prob={}".format(seq_out[i,:]))
            # print("normalized={}".format(normalized_vars_predict))
            maxi = np.argmax(normalized_vars_predict, axis=0)
            sampled_loc = self.random_sample_with_majority_voting(normalized_vars_predict)
            generated_routes.append(sampled_loc)
            # print("maxi {}, sampled maxi: {}".format(maxi, sampled_loc))
            # print("maxi={}".format(maxi))

            ## transforms to the next state
            next_state = self.state_next[state[i]]

            state[i + 1] = np.take(next_state, sampled_loc)
            # print('state='+str( state[:,i+1] ))
            # update to_visit, visited, loc, time
            to_visit[sampled_loc] = 0.
            visited[sampled_loc] = 1.
            loc[i + 1] = sampled_loc

        return generated_routes


    @staticmethod
    def generate_route_without_crisp(self, seq_out, visit):
        max_stop_seq_out, n_batch, n_locs = seq_out.shape
        seq_out = seq_out.squeeze()
        assert max_stop_seq_out == self.max_stops + 1
        assert n_locs == self.n_locs
        # to visit
        to_visit = copy.deepcopy(visit).detach().numpy().flatten()
        generated_routes = []
        for i in range(self.max_stops + 1):
            current_out = np.multiply(seq_out[i, :], to_visit)
            normalized_vars_predict = current_out / np.sum(current_out, keepdims=True)
            normalized_vars_predict[np.isnan(normalized_vars_predict)] = 0.0
            sampled_loc = self.random_sample_with_majority_voting(normalized_vars_predict)
            generated_routes.append(sampled_loc)
        return generated_routes




    def random_sample_with_majority_voting(self,probs, num_of_tryouts=100):
        probs = probs.flatten()
        ranges = self.convert_prob_to_range(probs)
        random_zs = np.random.random(num_of_tryouts)
        majority_vote = defaultdict(int)
        for z in random_zs:
            picked_loc = self.get_location_from_prob_range(ranges, z)
            majority_vote[picked_loc] += 1

        sort_majority_vote = sorted(majority_vote.items(), key=lambda x: x[1], reverse=True)
        return sort_majority_vote[0][0]


    def convert_prob_to_range(self, probs):
        sumed = 0.
        histgram = [0., ]
        for p in probs:
            sumed += p
            histgram.append(sumed)
        ranges = []
        for i in range(len(histgram)-1):
            ranges.append((histgram[i], histgram[i+1]))
        return ranges


    def get_location_from_prob_range(self,ranges, z):
        for i, (left, right) in enumerate(ranges):
            if left == right:
                continue
            if left <= z < right:
                return i
        return 0




