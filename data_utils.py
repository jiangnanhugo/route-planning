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

class ScheduleDataGen(object):
    def __init__(self, data_file, max_stop, num_locs):
        self.data = []
        self.rewards = []
        fr = open(data_file, 'r')
        for line in fr.readlines():
            tt = line[:-1].split()
            num_loc = int(tt[0])
            trajectory = [int(x) for x in tt[1:(num_loc+1)]]
            self.data.append(trajectory)
            self.rewards.append(float(tt[num_loc+1]))

        self.max_stop = max_stop + 1
        self.num_locs = num_locs
        self.num_data = len(self.data)

        self.it = 0
        # NOTICE: No need to randomize the data, the data is already randomized!

    def next_data(self, batch_size):
        """
        :param batch_size: batch size
        :return:
        """
        data = np.zeros((self.max_stop, batch_size, self.num_locs))
        visit = np.zeros((batch_size, self.num_locs))
        paths = np.zeros((self.max_stop, batch_size), dtype=np.int)
        daily_requests = []
        for i in range(batch_size):
            it = (self.it + i) % self.num_data
            j = 0
            one_request = set([])
            while j < len(self.data[it]):
                loc = self.data[it][j]
                visit[i, loc] = 1.0
                data[j, i, loc] = 1.0
                paths[j, i] = loc
                j += 1
                one_request.add(loc)
            # append to the same length
            while j < self.max_stop:
                data[j, i, loc] = 1.0
                paths[j, i] = loc
                j += 1
                one_request.add(loc)
            daily_requests.append(one_request)

        self.it += batch_size
        self.it %= self.num_data

        return data, visit, paths, daily_requests


def score_to_routes(score, real_path_canddates, use_post_process_type2):
    max_stops, nlocs = score.shape
    predict_routes = []
    mask = np.zeros(nlocs)
    for x in real_path_canddates:
        mask[x] = 1.0
    for i in range(max_stops):
        # print(score[i, :])
        if use_post_process_type2:
            loc = np.argmax(score[i, :] * mask)
        else:
            loc = np.argmax(score[i, :])
        predict_routes.append(loc)
    return predict_routes
