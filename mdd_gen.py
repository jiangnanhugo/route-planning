import copy
import numpy as np


class MDDTSPMask(object):
    def __init__(self, mdd):
        self.mdd = mdd

        self.n_states = len(self.mdd.mdd)
        self.n_locs = self.mdd.n_locations

        self.max_duration = self.mdd.max_duration
        self.distance_matrix = np.array(self.mdd.distance_matrix)
        self.max_stops = self.mdd.max_stops
        self.startp = self.mdd.startp
        self.endp = self.mdd.endp


        ### state_next
        ### state_mask
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

    def generate_mask(self, seq_out, visit):
        max_stop_seq_out, n_batch, n_locs = seq_out.shape
        assert max_stop_seq_out == self.max_stops + 1
        assert n_locs == self.n_locs

        # state
        state = np.zeros((n_batch, self.max_stops + 2), dtype=np.int32)
        state[:, 0] = self.mdd.root

        # loc
        loc=np.zeros((n_batch, self.max_stops + 2), dtype=np.int32)
        loc[:, 0] = self.startp

        # time
        time=np.zeros((n_batch, self.max_stops + 2))
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
            out_mask0 = np.multiply(np.multiply(1.0 - visited, mask), to_visit)
            out_mask = out_mask0.clone()

            for j in range(n_batch):
                for a in range(self.n_locs):
                    if out_mask0[j,a] > 0:

                        # time filter
                        if time[j,i] + self.distance_matrix[loc[j, i], a] + self.state_latest_time[state[j, i], a] > self.max_duration:
                            out_mask[j,a] = 0.

                        # some down filter
                        next_state = self.state_next[state[j, i], a]
                        for locid in range(self.n_locs):
                            if locid != a and to_visit[j, locid] > 0 and self.state_some_up[next_state,locid] == 0:
                                out_mask[j, a] = 0.
                                break

            out_mask_all[i,:,:] = out_mask

            current_out = np.multiply(seq_out[i], out_mask) - 1.0 + out_mask
            #current_out=seq_out[i].masked_fill(out_mask == 0, -np.inf)

            maxi = np.argmax(current_out, axis=1)

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
