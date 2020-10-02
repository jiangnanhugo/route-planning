
class MDDNode(object):
    """MDDNode represents a single node in the MDD.  An MDDNode is uniquely
    identified by its layer and state.  The (node) state must be a hashable
    object.

    Args:
        layer (int): layer the node is in
        state (object): state associated with node
    """
    def __init__(self, layer, state):
        """Construct a new 'MDDNode' object."""
        self.layer = layer
        self.state = state

    # MDDNodes to be used as dictionary keys.
    def __hash__(self):
        """Return the hash value."""
        return hash((self.layer, self.state))

    # Rich comparison methods: here the latter four are automatically
    # derived from the first two
    def __eq__(self, other):
        """Return self == other."""
        return self.layer == other.layer and self.state == other.state

    def __lt__(self, other):
        """Return self < other."""
        if self.layer != other.layer:
            return self.layer < other.layer
        else:
            return self.state < other.state

    def __ne__(self, other):
        """Return self != other."""
        return not self.__eq__(other)

    def __gt__(self, other):
        """Return self > other."""
        return not(self.__eq__(other) or self.__lt__(other))

    def __le__(self, other):
        """Return self <= other."""
        return self.__eq__(other) or self.__lt__(other)

    def __ge__(self, other):
        """Return self >= other."""
        return self.__eq__(other) or not self.__lt__(other)

    def __str__(self):
        return 'Node(' + str(self.layer) + ', ' + str(self.state) + ')'

    def __repr__(self):
        return 'Node(' + repr(self.layer) + ', ' + repr(self.state) + ')'


class MDDNodeInfo(object):
    """MDDNodeInfo represents information associated with an MDDNode.

    Args:
        incoming (set): set of incoming arcs (default: set())
        outgoing (set): set of outgoing arcs (default: set())
    """

    def __init__(self, incoming=None, outgoing=None):
        """Construct a new 'MDDNode' object."""
        if incoming is None:
            self.incoming = set()
        if outgoing is None:
            self.outgoing = set()

    def __str__(self):
        return '<in=' + str(self.incoming) + ', out=' + str(self.outgoing) + '>'

    def __repr__(self):
        return '<in=' + repr(self.incoming) + ', out=' + repr(self.outgoing) + '>'
