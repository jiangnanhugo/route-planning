import numpy as np


class ScheduleProb(object):
    def __init__(self):
        self.initialized = False

    def init_by_assign(self, paired_dist, startp, endp, max_duration, max_stops, rewards):
        self.paired_dist = paired_dist
        self.startp = startp
        self.endp = endp
        self.max_duration = max_duration
        self.max_stops = max_stops
        self.rewards = rewards
        self.num_locs = len(self.paired_dist)
        self.initialized = True

    def print_prob(self, oup):
        print('[schedule ins] paired_dist', self.paired_dist, file=oup)
        print('[schedule ins] startp', self.startp, file=oup)
        print('[schedule ins] endp', self.endp, file=oup)
        print('[schedule ins] max_duration', self.max_duration, file=oup)
        print('[schedule ins] max_stops', self.max_stops, file=oup)
        print('[schedule ins] rewards', self.rewards, file=oup)
        print('[schedule ins] num_locs', self.num_locs, file=oup)

    def norm_reward(self, syn, real):
        total_reward = 0.
        for si, ri in zip(syn, real):
            total_reward += self.reward(si) * 1.0 / self.reward(ri)
        return total_reward / len(syn)

    def reward(self, oup):
        reward = 0.
        for i, loc in enumerate(oup):
            reward += self.rewards[i][loc]
        return reward

    def valid_routes(self, oup, visit):
        num_valid = np.zeros(len(oup), dtype=np.int32)
        num_loc_invalid = 0
        num_dis_invalid = 0
        num_strict_subset = 0
        num_empty = 0

        for idx, (oi, vi) in enumerate(zip(oup, visit)):
            dis = 0.
            visited = set([])
            to_visit = set([i for i in range(self.num_locs) if vi[i] > 0.5])
            cur_loc = self.startp

            loc_valid = True
            is_subset = True

            last_non_endp = -1
            first_endp = -1
            for i, next_loc in enumerate(oi):
                if (next_loc in visited) or (not next_loc in to_visit):
                    loc_valid = False

                if (not next_loc in to_visit) and (not next_loc in visited):
                    is_subset = False

                if next_loc != self.endp:
                    visited.add(next_loc)
                    if next_loc in to_visit:
                        to_visit.remove(next_loc)

                if next_loc == self.endp and first_endp < 0:
                    first_endp = i

                if next_loc != self.endp:
                    last_non_endp = i

                dis += self.paired_dist[cur_loc][next_loc]
                cur_loc = next_loc

            if last_non_endp >= first_endp:
                loc_valid = False

            if dis > self.max_duration:
                dis_valid = False
            else:
                dis_valid = True

            if not loc_valid:
                num_loc_invalid += 1

            if not dis_valid:
                num_dis_invalid += 1

            if (len(to_visit) != 1 or (not self.endp in to_visit)) and (is_subset) and dis_valid:
                if last_non_endp >= 0:
                    num_strict_subset += 1
                subset_valid = False
            else:
                subset_valid = True

            if subset_valid and loc_valid and dis_valid:
                num_valid[idx]= 1

            if last_non_endp < 0:
                num_empty += 1

        return num_valid


class ScheduleDataGen:
    def __init__(self, data_file, max_stop, num_locs):
        self.data = []
        self.rewards = []
        inp = open(data_file, 'r')
        for l in inp.readlines():
            tt = l[:-1].split()
            num_loc = int(tt[0])
            this_visit = [int(itm) for itm in tt[1:(num_loc+1)]]
            self.data.append(this_visit)
            self.rewards.append(float(tt[num_loc+1]))
        inp.close()

        # print('rewards=', self.rewards)
        # print('data=', self.data)

        self.max_stop = max_stop
        self.max_stop += 1
        self.num_locs = num_locs
        self.num_data = len(self.data)

        self.it = 0
        # NOTICE: No need to randomize the data, the data is already randomized!

    def next_data(self, batchnum):
        data = np.zeros((self.max_stop, batchnum, self.num_locs))
        visit = np.zeros((batchnum, self.num_locs))

        for i in range(batchnum):
            it = (self.it + i) % self.num_data
            j = 0
            while j < len(self.data[it]):
                loc = self.data[it][j]
                visit[i, loc] = 1.0
                data[j, i, loc] = 1.0
                j += 1
            while j < self.max_stop:
                data[j, i, loc] = 1.0
                j += 1

        self.it += batchnum
        self.it %= self.num_data

        return data, visit



