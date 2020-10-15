import copy
import numpy as np
from collections import defaultdict
from mdd.utils import get_mdd


class CRISP(object):
    def __init__(self, n_locations, max_stops, maxwidth):
        self.maxwidth = maxwidth
        self.n_locations = n_locations
        self.max_stops = max_stops
        self.mdd = get_mdd(n_locations, max_stops, maxwidth)

    def generate_mask_with_ground_truth(self, real_paths, daily_requests):
        """
        in training, convert the mdd into mask vector
        :param mdd:
        :param real_paths:  the trajectory
        :param daily_requests: the set of locations in array form. X_i=1 mean the location is in the daily request
        :return:
        """
        max_stop_seq_out, n_batch = real_paths.shape
        # [6, batch_size, 29]
        out_mask_all = np.zeros((self.max_stops + 1, n_batch, self.n_locations), dtype=np.float32)
        for i in range(n_batch):
            # filtered_mdd = self.mdd_filtering(self.mdd, daily_requests[i])
            neighbor_along_path = self.mdd.find_neighbor_along_path(real_paths[:, i])
            new_neighbor_along_path = self.fast_mdd_filtering(neighbor_along_path, daily_requests[i])
            out_mask_all[:, i, :] = self.idx_to_binary(new_neighbor_along_path)

        return out_mask_all

    def fast_mdd_filtering(self, neighbor_along_path, daily_request):
        new_neighbor_along_path = []
        for i, layer in enumerate(neighbor_along_path):
            new_layer = [x for x in layer if x in daily_request]
            new_neighbor_along_path.append(new_layer)
        return new_neighbor_along_path



    # MDD Filtering to Process Daily Requests
    def mdd_filtering(self, MDD, daily_request):
        """
        :param MDD: a graph
        :param daily_request: a set of locations. {1,2,3,4,0}
        :return:
        """
        filtered_mdd = copy.deepcopy(MDD)
        for j in range(filtered_mdd.numArcLayers):
            nodes_in_layer = [v for v in filtered_mdd.allnodes_in_layer(j)]
            for v in nodes_in_layer:
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
        mask = np.zeros(shape=(self.max_stops + 1, self.n_locations), dtype=np.float)
        for i, layer in enumerate(neighbor_along_paths):
            for x in layer:
                mask[i, x] = 1.0
        return mask

    def valid_routes(self, oups, daily_requests):
        num_valid = 0
        for oup, daily_request in zip(oups, daily_requests):
            full_conver = True
            all_diff = True
            visited = set()
            for x in oup:
                if x not in daily_request:
                    full_conver = False
                if x in visited:
                    all_diff = False
                if x != 0:
                    visited.add(x)
            num_valid += full_conver and all_diff

        return num_valid

    def generate_route_for_inference(self, seq_out, daily_requests):
        max_stop_seq_out, n_batch, n_locs = seq_out.shape
        seq_out = seq_out.squeeze()
        assert max_stop_seq_out == self.max_stops + 1
        assert n_locs == self.n_locations

        generated_routes = []
        for nb in range(n_batch):
            generated_route = []
            filtered_mdd = self.mdd_filtering(self.mdd, daily_requests[nb])
            for i in range(len(daily_requests[nb])):
                # [6, batch_size, 29]
                neighbor_along_path = filtered_mdd.find_last_neighbor_along_path(generated_route)

                out_mask = self.idx_to_binary(neighbor_along_path)

                current_out = np.multiply(seq_out[i, nb, :], out_mask[i, :])
                normalized_vars_predict = current_out / np.sum(current_out, keepdims=True)
                normalized_vars_predict[np.isnan(normalized_vars_predict)] = 0.0

                sampled_loc = self.random_sample_with_majority_voting(normalized_vars_predict)
                # print("{}-th sampled loc: {}".format(i, sampled_loc))
                generated_route.append(sampled_loc)
            generated_routes.append(generated_route)
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

    def random_sample_with_majority_voting(self, probs, num_of_tryouts=100):
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
        for i in range(len(histgram) - 1):
            ranges.append((histgram[i], histgram[i + 1]))
        return ranges

    def get_location_from_prob_range(self, ranges, z):
        for i, (left, right) in enumerate(ranges):
            if left == right:
                continue
            if left <= z < right:
                return i
        return 0
