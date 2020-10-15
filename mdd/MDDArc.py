
class MDDArc(object):
    """MDDArc represents a single arc in the MDD.  An MDDArc is uniquely
    identified by its head/tail nodes, label.

    Args:
        label (object): label of arc (e.g., assigned value)
        tail (MDDNode): tail/source node
        head (MDDNode): head/destination node
    """

    def __init__(self, label, tail, head):
        """Construct a new 'MDDArc' object."""
        self.label = label
        self.tail = tail
        self.head = head

    # Allows MDDArcs to be used as dictionary keys.
    def __hash__(self):
        """Return the hash value."""
        return hash((self.label, self.tail, self.head))

    # Rich comparison methods: here the latter four are automatically
    # derived from the first two
    def __eq__(self, other):
        """Return self == other."""
        return self.tail == other.tail and self.label == other.label and self.head == other.head
    #
    # def __lt__(self, other):
    #     """Return self < other."""
    #     if self.tail != other.tail:
    #         return self.tail < other.tail
    #     elif self.label != other.label:
    #         return self.label < other.label
    #     elif self.head != other.head:
    #         return self.head < other.head

    def __ne__(self, other):
        """Return self != other."""
        return not self.__eq__(other)

    # def __gt__(self, other):
    #     """Return self > other."""
    #     return not(self.__eq__(other) or self.__lt__(other))

    # def __le__(self, other):
    #     """Return self <= other."""
    #     return self.__eq__(other) or self.__lt__(other)
    #
    # def __ge__(self, other):
    #     """Return self >= other."""
    #     return self.__eq__(other) or not self.__lt__(other)

    def __str__(self):
        return 'Arc(' + str(self.label) + ': ' + str(self.tail) + ', ' + str(self.head) + ')'
    #
    # def __repr__(self):
    #     return 'Arc(' + repr(self.label) + ', ' + repr(self.tail) + ', ' + repr(self.head) + ')'
