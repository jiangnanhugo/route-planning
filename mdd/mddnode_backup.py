from hashlib import blake2b
import copy


def is_same(a, b):
    if len(a) != len(b):
        return False
    else:
        for i, j in zip(a, b):
            if i != j:
                return False
        return True

class MDDHash(object):
    def __init__(self, mdd, _hashsize=1000003):
        self.hashsize = _hashsize
        self.h = [[] for _ in range(self.hashsize)]
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
            if is_same(a, self.mdd[itm].incoming) and is_same(n, self.mdd[itm].outgoing):
                return idx, i, itm
        return None, None, None

    def insert(self, a, n, mdd_idx):
        idx = self.encode(a, n)
        self.h[idx].append(mdd_idx)

    def delete(self, h_idx, h_it):
        return self.h[h_idx].pop([h_it])

class MDDNode(object):
    def __init__(self, _a, _n):
        """
        :param _a: list of incoming arcs. previous arc
        :param _n: list of outgoing arcs. next arc
        :param count: the number of incoming arcs. If count=0, means this node have no incoming arc.
        """
        self.incoming = copy.copy(_a)
        self.outgoing = copy.copy(_n)
        self.count = 0

    def num(self):
        return len(self.outgoing)

    def __del__(self):
        print("object deleted")
        del self.incoming
        del self.outgoing



def get_line_start_with(inp, start_line):
    l = inp.readline()
    while l != "" and not l.startswith(start_line):
        l = inp.readline()