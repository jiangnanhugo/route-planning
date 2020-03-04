from hashlib import blake2b
import copy


def is_same(a, b):
    if len(a) != len(b):
        return False
    else:
        for i,j in zip(a, b):
            if i != j:
                return False
        return True


class MDDNode:
    def __init__(self, _a=[], _n=[]):
        self.a = copy.copy(_a)
        self.n = copy.copy(_n)

        self.earliest_time = []
        self.latest_time = []

        self.all_up = set([])
        self.some_up = set([])
        self.all_down = set([])
        self.some_down = set([])

        self.count = 0

    def num(self):
        return len(self.n)


class MDDHash:
    def __init__(self, mdd, _hashsize=1000003):
        self.hashsize = _hashsize
        self.h = [[] for i in range(self.hashsize)]
        self.mdd = mdd

    def encode(self, a, n):
        an = []
        for itm in a:
            for i in range(4):
                an.append(itm % 256)
                itm >>= 8
        for itm in n:
            for i in range(4):
                an.append(itm % 256)
                itm >>= 8
        bb = int(blake2b(bytes(an), digest_size=4).hexdigest(), 16)
        return bb % self.hashsize

    def find(self, a, n):
        idx = self.encode(a, n)
        for i, itm in enumerate(self.h[idx]):
            if is_same(a, self.mdd[itm].a) and is_same(n, self.mdd[itm].n):
                return idx, i, itm
        return None, None, None

    def insert(self, a, n, mdd_idx):
        idx = self.encode(a, n)
        self.h[idx].append(mdd_idx)

    def delete(self, h_idx, h_it, mdd_idx):
        return self.h[h_idx].pop([h_it])


class MDD_TSP(object):
    def __init__(self, distance_matrix, startp, endp, max_duration, max_stops, max_width):
        self.distance_matrix = distance_matrix
        self.startp = startp
        self.endp = endp
        self.max_duration = max_duration
        self.max_stops = max_stops
        self.max_width = max_width

        self.n_locations = len(distance_matrix)

        self.mdd = []
        self.mdd_hash = MDDHash(self.mdd)

        self.layers = []

        ### build MDD

        # end point
        end_point = MDDNode([], [])
        self.mdd.append(end_point)
        self.mdd_hash.insert(end_point.a, end_point.n, 0)
        self.layers.append([0])

        # last stop
        last_stop = MDDNode([self.endp], [0])
        self.mdd.append(last_stop)
        self.mdd_hash.insert(last_stop.a, last_stop.n, 1)
        self.layers.append([1])
        self.mdd[0].count += 1

        self.len_mdd = 2

        # add others
        if self.startp == self.endp:
            minus = 1
        else:
            minus = 2
        for i in range(min(max_stops, self.n_locations - minus)):
            a = [i for i in range(self.n_locations) if (i != self.startp and i != self.endp)]
            n = [self.len_mdd-1 for i in range(self.n_locations) if (i != self.startp and i != self.endp)]
            #
            a.append(self.startp)
            n.append(0)

            this_stop = MDDNode(a, n)
            self.mdd.append(this_stop)
            self.mdd_hash.insert(a, n, self.len_mdd)
            self.layers.append([self.len_mdd])

            self.mdd[self.len_mdd-1].count += len(a) - 1
            self.mdd[0].count += 1
            self.len_mdd += 1

        self.mdd[self.len_mdd-1].count += 1
        self.root = self.len_mdd - 1
        self.bottom = 0

        self.layers.reverse()

    def earliest_time(self):
        for k in range(len(self.layers)):
            self.earliest_time_by_layer(k)

    def earliest_time_by_layer(self, k):
        if k == 0:
            node_id = self.layers[k][0]
            self.mdd[node_id].earliest_time = [self.distance_matrix[self.startp][ai] for i, ai in enumerate(self.mdd[node_id].a)]
            return

        for node_id in self.layers[k-1]:
            assert len(self.mdd[node_id].earliest_time) == len(self.mdd[node_id].a)

        for node_id in self.layers[k]:
            self.mdd[node_id].earliest_time = [float(0x7fffffff) for i in self.mdd[node_id].a]

        for node_id in self.layers[k-1]:
            node = self.mdd[node_id]
            for i0, a0 in enumerate(node.a):
                node_id1 = node.n[i0]
                node_next = self.mdd[node_id1]
                earliest_node = node.earliest_time[i0]
                for i1, a1 in enumerate(node_next.a):
                    node_next.earliest_time[i1] = min(earliest_node + self.distance_matrix[a0][a1],
                                                      node_next.earliest_time[i1])

    def latest_time(self):
        for k in range(len(self.layers) - 2, -1, -1):
            self.latest_time_by_layer(k)

    def latest_time_by_layer(self, k):
        if k == len(self.layers) - 2:
            node_id = self.layers[k][0]
            self.mdd[node_id].latest_time = [0.0 for ai in self.mdd[node_id].a]
            return

        for node_id in self.layers[k+1]:
            assert len(self.mdd[node_id].latest_time) == len(self.mdd[node_id].a)

        for node_id in self.layers[k]:
            self.mdd[node_id].latest_time = [float(0x7fffffff) for i in self.mdd[node_id].a]

        for node_id in self.layers[k]:
            node = self.mdd[node_id]
            for i, ai in enumerate(node.a):
                if ai == self.endp:
                    node.latest_time[i] = 0.

        for node_id in self.layers[k]:
            node = self.mdd[node_id]
            for i0, a0 in enumerate(node.a):
                node_id1 = node.n[i0]
                node_next = self.mdd[node_id1]
                for i1, a1 in enumerate(node_next.a):
                    node.latest_time[i0] = min(node_next.latest_time[i1] + self.distance_matrix[a0][a1],
                                               node.latest_time[i0])

    def all_up_by_layer(self, k):
        # print('k=', k)
        if k == len(self.layers) - 1:
            node_id = self.layers[k][0]
            self.mdd[node_id].all_up = set([])
            return

        for node_id in self.layers[k+1]:
            assert not -1 in self.mdd[node_id].all_up

        for node_id in self.layers[k]:
            self.mdd[node_id].all_up.add(-1)

        for node_id in self.layers[k]:
            node = self.mdd[node_id]
            for a, n in zip(node.a, node.n):
                node_next = self.mdd[n]
                if -1 in node.all_up:
                    node.all_up = node_next.all_up | set([a])
                else:
                    node.all_up &= (node_next.all_up | set([a]))

    def all_up(self):
        for k in range(len(self.layers) - 2, -1, -1):
            self.all_up_by_layer(k)


    def some_up_by_layer(self, k):
        # print('k=', k)
        if k == len(self.layers) - 1:
            node_id = self.layers[k][0]
            self.mdd[node_id].some_up = set([])
            return

        for node_id in self.layers[k]:
            node = self.mdd[node_id]
            node.some_up = set([])

        for node_id in self.layers[k]:
            node = self.mdd[node_id]
            for a, n in zip(node.a, node.n):
                node_next = self.mdd[n]
                node.some_up |= node_next.some_up | set([a])


    def some_up(self):
        for k in range(len(self.layers) - 2, -1, -1):
            self.some_up_by_layer(k)

    def all_down(self):
        for k in range(len(self.layers)):
            self.all_down_by_layer(k)

    def all_down_by_layer(self, k):
        if k == 0:
            node_id = self.layers[k][0]
            self.mdd[node_id].all_down = set([])
            return

        for node_id in self.layers[k-1]:
            assert not -1 in self.mdd[node_id].all_down

        for node_id in self.layers[k]:
            self.mdd[node_id].all_down.add(-1)

        for node_id in self.layers[k-1]:
            node = self.mdd[node_id]
            for a, n in zip(node.a, node.n):
                node_next = self.mdd[n]
                if -1 in node_next.all_down:
                    node_next.all_down = node.all_down | set([a])
                else:
                    node_next.all_down &= (node.all_down | set([a]))


    def some_down_by_layer(self, k):
        if k == 0:
            node_id = self.layers[k][0]
            self.mdd[node_id].some_down = set([])
            return

        for node_id in self.layers[k]:
            node = self.mdd[node_id]
            node.some_down = set([])

        for node_id in self.layers[k-1]:
            node = self.mdd[node_id]
            for a, n in zip(node.a, node.n):
                node_next = self.mdd[n]
                node_next.some_down |= (node.some_down | set([a]))


    def some_down(self):
        for k in range(len(self.layers)):
            self.some_down_by_layer(k)


    def delete_empty_nodes_by_layer(self, k):
        check_previous_layer = False
        idx = 0
        while idx < len(self.layers[k]):
            node_id = self.layers[k][idx]
            node = self.mdd[node_id]
            if len(node.a) == 0:
                check_previous_layer = True
                del self.layers[k][idx]
                if k > 0:
                    for pre_node_id in self.layers[k-1]:
                        pre_node = self.mdd[pre_node_id]
                        i = 0
                        while i < len(pre_node.n):
                            if pre_node.n[i] == node_id:
                                del pre_node.a[i]
                                del pre_node.n[i]
                                del pre_node.earliest_time[i]
                                del pre_node.latest_time[i]
                            else:
                                i += 1
            else:
                idx += 1

        if check_previous_layer and k > 0:
            self.delete_empty_nodes_by_layer(k-1)

    def filter_by_layer(self, k):
        # can affect all_up, some_up, all_down, some_down

        delete_layer_check = False
        for node_id in self.layers[k]:
            node = self.mdd[node_id]

            i = 0
            while i < len(node.a):
                node_next = self.mdd[node.n[i]]
                if (node.earliest_time[i] + node.latest_time[i] > \
                   self.max_duration):
                    delete_layer_check = True
                    node_next.count -= 1
                    del node.a[i]
                    del node.n[i]
                    del node.earliest_time[i]
                    del node.latest_time[i]
                # elif node.a[i] == self.startp or node.a[i] == self.endp:
                #     i += 1
                elif (node.a[i] in node.all_down) or \
                     (node.a[i] in node_next.all_up) or \
                     (k == len(node.some_down) and node.a[i] in node.some_down) or \
                     (k == len(node_next.some_up) and node.a[i] in node_next.some_up):
                    delete_layer_check = True
                    node_next.count -= 1
                    del node.a[i]
                    del node.n[i]
                    del node.earliest_time[i]
                    del node.latest_time[i]
                else:
                    i += 1

        if delete_layer_check:
            self.delete_empty_nodes_by_layer(k)

    def refine_by_layer(self, k):
        while len(self.layers[k]) < self.max_width:
            # find a node that can be splitted
            split_node_found = False
            for pre_node_id in self.layers[k-1]:
                pre_node = self.mdd[pre_node_id]
                for i, ai in enumerate(pre_node.a):
                    ni = pre_node.n[i]
                    node = self.mdd[ni]
                    if ai in (node.some_down - node.all_down):
                        split_node_found = True
                        break
                if split_node_found:
                    break

            if not split_node_found:
                break
            else:
                new_node = MDDNode(node.a, node.n)
                for n in node.n:
                    self.mdd[n].count += 1

                new_node.latest_time = copy.copy(node.latest_time)
                new_node.all_up = copy.copy(node.all_up)
                new_node.some_up = copy.copy(node.some_up)

                self.mdd.append(new_node)
                self.layers[k].append(self.len_mdd)
                self.len_mdd += 1

                new_node.all_down = set([-1])
                node.all_down = set([-1])
                new_node.some_down = set([])
                node.some_down = set([])

                node.count = 0

                # old_node_edge_poll = pre_node.all_down | set([ai])

                for pre_node_id in self.layers[k-1]:
                    pre_node = self.mdd[pre_node_id]
                    for i1 in range(len(pre_node.n)):
                        if pre_node.n[i1] == ni:
                            #if pre_node.a[i1] in old_node_edge_poll:
                            if (ai in pre_node.all_down) or (ai == pre_node.a[i1]):
                                pre_node.n[i1] = ni
                                node.count += 1

                                # all down
                                if -1 in node.all_down:
                                    node.all_down = pre_node.all_down | set([pre_node.a[i1]])
                                else:
                                    node.all_down &= pre_node.all_down | set([pre_node.a[i1]])
                                # some down
                                node.some_down |= pre_node.some_down | set([pre_node.a[i1]])
                            else:
                                pre_node.n[i1] = self.len_mdd - 1
                                new_node.count += 1

                                # all down
                                if -1 in new_node.all_down:
                                    new_node.all_down = pre_node.all_down | set([pre_node.a[i1]])
                                else:
                                    new_node.all_down &= pre_node.all_down | set([pre_node.a[i1]])
                                # some down
                                new_node.some_down |= pre_node.some_down | set([pre_node.a[i1]])

    def remove_from_layer_zero_count(self):
        for layer in self.layers:
            idx = 0
            while idx < len(layer):
                if self.mdd[layer[idx]].count <= 0:
                    del layer[idx]
                else:
                    idx += 1


    def filter_refine(self):
        for k in range(0, len(self.layers) - 2):
            # delete all the nodes from layer k

            self.filter_by_layer(k)

            self.all_down_by_layer(k+1)
            self.some_down_by_layer(k+1)
            self.all_up_by_layer(k)
            self.some_up_by_layer(k)

            self.refine_by_layer(k+1)

            self.earliest_time_by_layer(k+1)

        self.remove_from_layer_zero_count()

    def filter_refine_preparation(self):
        self.earliest_time()
        self.latest_time()

        self.all_down()
        self.some_down()
        self.all_up()
        self.some_up()

    def add_last_node_forever(self):
        self.mdd[0].a.append(self.endp)
        self.mdd[0].n.append(0)
        self.mdd[0].some_up.add(self.endp)
        self.mdd[0].all_up.add(self.endp)


    def print_mdd(self, oup):
        # oup = open(out_file, 'w')
        print('NumLayer', len(self.layers), file=oup)
        for layer in self.layers:
            oup.write('L '+str(len(layer)))
            for node in layer:
                oup.write(' '+str(node))
            oup.write('\n')
            for node_id in layer:
                node = self.mdd[node_id]
                print('N', node_id, node.count, node.num(), file=oup)

                oup.write('a')
                for a in node.a:
                    oup.write(' '+str(a))
                oup.write('\n')

                oup.write('n')
                for n in node.n:
                    oup.write(' '+str(n))
                oup.write('\n')

                oup.write('e')
                for e in node.earliest_time:
                    oup.write(' '+str(e))
                oup.write('\n')

                oup.write('l')
                for l in node.latest_time:
                    oup.write(' '+str(l))
                oup.write('\n')

                oup.write('ad')
                for l in node.all_down:
                    oup.write(' '+str(l))
                oup.write('\n')

                oup.write('sd')
                for l in node.some_down:
                    oup.write(' '+str(l))
                oup.write('\n')

                oup.write('au')
                for l in node.all_up:
                    oup.write(' '+str(l))
                oup.write('\n')

                oup.write('su')
                for l in node.some_up:
                    oup.write(' '+str(l))
                oup.write('\n')
        # oup.close()


def get_line_start_with(inp, start_line):
    l = inp.readline()
    while l != "" and not l.startswith(start_line):
        l = inp.readline()
