from random import shuffle
from mdd.mddnode_backup import *

# total travel distance
# exact
# relaxed. fig1.
# for node u4 the earlist time is: the shortest distance.
# for node u4 the latest time is: the longest distance.
# if all the path from t to u4 pass l2. then l2 will be removed for all the out

# some constraints can be ignored. only need to proof of our concept.

class MDD_TSP(object):
    def __init__(self, distance_matrix, startp, endp, max_stops):
        """
        :param distance_matrix:
        :param startp: start position
        :param endp: end position
        :param max_stops:
        """
        self.dist_matrix = distance_matrix
        self.startp = startp
        self.endp = endp
        self.max_stops = max_stops

        self.n_locations = len(distance_matrix)

        self.mdd = []
        self.mdd_hash = MDDHash(self.mdd)
        self.layers = []

        ### build MDD
        # end point
        end_point = MDDNode([], [])
        self.mdd.append(end_point)
        self.mdd_hash.insert(end_point.incoming, end_point.outgoing, 0)
        self.layers.append([0])

        # last stop
        last_stop = MDDNode([self.endp], [0])
        self.mdd.append(last_stop)
        self.mdd_hash.insert(last_stop.incoming, last_stop.outgoing, 1)
        self.layers.append([1])

        self.len_mdd = 2

        # add others
        minus = 2
        if self.startp == self.endp:
            minus = 1
        for i in range(min(max_stops, self.n_locations - minus)):
            incoming = [i for i in range(self.n_locations) if (i != self.startp and i != self.endp)]
            outgoing = [self.len_mdd-1 for i in range(self.n_locations) if (i != self.startp and i != self.endp)]
            #
            incoming.append(self.startp)
            outgoing.append(0)

            this_stop = MDDNode(incoming, outgoing)
            self.mdd.append(this_stop)
            self.mdd_hash.insert(incoming, outgoing, self.len_mdd)
            self.layers.append([self.len_mdd])

            self.mdd[self.len_mdd-1].count += len(incoming) - 1
            self.mdd[0].count += 1
            self.len_mdd += 1

        self.mdd[self.len_mdd-1].count += 1
        self.root = self.len_mdd - 1
        self.bottom = 0

        self.layers.reverse()

    def arc_filtering_by_layer(self, k):
        """ arc filtering
        :param k: k-th layer
        """
        for node_id in self.layers[k]:
            node = self.mdd[node_id]
            i = 0
            while i < len(node.incoming):
                node_next = self.mdd[node.outgoing[i]]
                if node.incoming:
                    node_next.count -= 1
                    del node.incoming[i]
                    del node.outgoing[i]
                    print("delete pre_node={} because of max duration".format(node))
                else:
                    i += 1

    def node_splitting_by_layer(self, k, max_width):
        """ node splitting
        :param k: the k-th layer
        """
        while len(self.layers[k]) < max_width:
            print("{} -th layer is larger than the maximum width".format(k))

            split_node_found = False
            node = None
            ni = None
            ai = None
            for pre_node_id in self.layers[k-1]:                    # find a node that can be split
                pre_node = self.mdd[pre_node_id]
                for i, ai in enumerate(pre_node.incoming):
                    ni = pre_node.outgoing[i]
                    node = self.mdd[ni]
                    if len(node.incoming) > 1:
                        split_node_found = True
                        break
                if split_node_found:
                    break

            if not split_node_found:
                break
            """
            split the node into two sub nodes
            """
            shuffled_incoming = shuffle(node.incoming)
            income_part1, income_part2 = shuffled_incoming[:len(shuffled_incoming)//2], shuffled_incoming[len(shuffled_incoming)//2:]
            new_node1 = MDDNode(income_part1, node.outgoing)       # new node is a replicate of the old one
            new_node2 = MDDNode(income_part2, node.outgoing)       # new node is a replicate of the old one
            self.mdd.append(new_node1)                              # add the new node into the mdd
            self.layers[k].append(self.len_mdd)
            self.len_mdd += 1
            self.mdd.append(new_node2)  # add the new node into the mdd
            self.layers[k].append(self.len_mdd)
            self.len_mdd += 1


            for pre_node_id in self.layers[k - 1]:
                pre_node = self.mdd[pre_node_id]
                for j in range(len(pre_node.outgoing)):
                    if pre_node.outgoing[j] == ni:
                        if ai == pre_node.incoming[j]:
                            pre_node.outgoing[j] = ni
                        else:
                            pre_node.outgoing[j] = self.len_mdd - 1

    def remove_from_layer_zero_count(self):
        for layer in self.layers:
            idx = 0
            while idx < len(layer):
                if self.mdd[layer[idx]].count <= 0:
                    del layer[idx]
                else:
                    idx += 1

    def relax_mdd(self, max_width):
        for k in range(0, len(self.layers) - 2):
            # delete all the nodes from layer k
            self.node_splitting_by_layer(k + 1, max_width=max_width)
            self.arc_filtering_by_layer(k)

    def add_last_node_forever(self):
        self.mdd[0].incoming.append(self.endp)
        self.mdd[0].outgoing.append(0)

    def print_mdd(self, oup):
        print('NumLayer', len(self.layers), file=oup)
        for layer in self.layers:
            oup.write('L '+str(len(layer)))
            for node in layer:
                oup.write(' '+str(node))
            oup.write('\n')
            for node_id in layer:
                node = self.mdd[node_id]
                print('N', node_id, node.count, node.num(), file=oup)

                oup.write('income')
                for income in node.incoming:
                    oup.write(' '+str(income))
                oup.write('\n')

                oup.write('out')
                for out in node.outgoing:
                    oup.write(' '+str(out))
                oup.write('\n')