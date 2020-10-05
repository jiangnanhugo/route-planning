from mdd.mdd import MDD
import random


def get_idx(temp_list, nodename, is_income=True):
    for i in range(len(temp_list)):
        if is_income and temp_list[i].in_come == nodename:
            return i
        if (not is_income) and temp_list[i].out_go == nodename:
            return i
    return -1


class Node(object):
    def __init__(self, layer, state, in_come, out_go):
        self.layer = layer

        def get_hash():
            return random.getrandbits(40)
        self.id = get_hash()
        self.Type = "node"
        self.state = state
        self.in_come = in_come
        self.out_go = out_go

    def get_string(self):
        return {'state': self.state, 'id': self.id,
                'layer': self.layer, 'Type': self.Type,
                'in_come': self.in_come, 'out_go': self.out_go}

    def __str__(self):
        return '{state: ' + self.state + ', id:' + str(self.id) + ', layer:' + str(self.layer) + \
               ', Type:' + str(self.Type) + ", in_:" + self.in_come + ', out_:' + self.out_go + '}'


class Arc(object):
    def __init__(self, head_id, tail_id, name):
        self.Type = "arc"
        self.label = name
        self.head = head_id
        self.tail = tail_id

    def get_string(self):
        return {'label': self.label, 'head': self.head,
                'tail': self.tail, 'Type': self.Type}

    def __str__(self):
        return '{type: ' + self.Type + ', label:' + self.label + \
               ', head:' + str(self.head) + ', tail:' + str(self.tail) + '}'

def build_full_tsp(n_locations, max_stops):
    node_list = []
    # 0-th layer
    uid = 0
    source = Node(0, "s", 'none', 'any')
    sink = Node(n_locations+1, "t", 'any', 't')
    uid += 1
    node_list.append(source)

    # middle layer
    for layer in range(max_stops):
        if layer == max_stops-1:
            new_node = Node(layer + 1, "u" + str(uid), "layer_"+str(layer), "t")
        else:
            new_node = Node(layer + 1, "u" + str(uid), "layer_"+str(layer), "layer_"+str(layer+1))
        node_list.append(new_node)
        uid += 1
    node_list.append(sink)

    # build arc
    arc_list = []
    for layer in range(max_stops):                                           # type 1 arc
        temp_list = []
        for i in range(n_locations):
            new_arc = Arc(node_list[layer+1].id, node_list[layer].id, i+1)
            temp_list.append(new_arc)
        arc_list.append(temp_list)

    for node in node_list[1:-1]:                                               # type 2 arc
        arc_list.append([Arc(node_list[-1].id, node.id, 0)])

    init = []
    for x in node_list:
        init.append(x.get_string())

    for i in range(len(arc_list)):
        for x in arc_list[i]:
            init.append(x.get_string())

    json_content = {"mdd_name": "TSP", "init": init}
    return json_content


def get_mdd(n_locations, max_stops, maxwidth=0):
    mymdd = MDD()
    json_content = build_full_tsp(n_locations, max_stops)
    mymdd.loadJSON(json_content)
    mymdd.relax_mdd(maxwidth)
    return mymdd

