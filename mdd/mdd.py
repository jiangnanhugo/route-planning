from itertools import chain # used in various places
from json import dump, load # used in dumpJSON and loadJSON
from collections import defaultdict
import copy
import numpy as np
from mdd.MDDArc import MDDArc
from mdd.MDDNode import MDDNode, MDDNodeInfo


class MDD(object):
    """MDD represents a multivalued decision diagram (MDD).
    Args:
        name (str): name of MDD (default: 'mdd')
        nodes (List[Dict[MDDNode, MDDNodeInfo]]): nodes of MDD;
            if None (default), set to empty list
    """
    def __init__(self, name='mdd', nodes=None):
        """Construct a new 'MDD' object."""
        # 'nodes' is a list of dicts (one for each node layer),
        # and each dict stores the nodes in that layer;
        # each node is represented as a (MDDNode, MDDNodeInfo) key-value pair
        self.nodes = nodes
        self.name = name
        if self.nodes is None:
            self.nodes = []

    @property
    def numNodeLayers(self):
        # Number of node layers; equal to number of 'variables' + 1.
        return len(self.nodes)

    @property
    def numArcLayers(self):
        # Number of arc layers; equal to number of 'variables'.
        return len(self.nodes)-1

    @property
    def widthList(self):
        # Number of nodes in each layer
        return list(len(lyr) for lyr in self.nodes)

    @property
    def maxWidth(self):
        # Maximum number of nodes in a single node layer.
        return max(len(lyr) for lyr in self.nodes)

    def __str__(self, showLong=False, showIncoming=False):
        """Return a (human-readable) string representation of the MDD.
        Args:
            showLong (bool): use more vertical space (default: False)
            showIncoming (bool): show incoming arcs (default: False)

        Returns:
            str: string representation of MDD
        """
        s = '== MDD (' + self.name + ', ' + str(self.numArcLayers) + ' layers) ==\n'
        if showLong:
            # Long form
            s += '# Nodes\n'
            for (j, lyr) in enumerate(self.nodes):
                s += 'Layer ' + str(j) + ':\n'
                for v in lyr:
                    s += '\t' + str(v) + ': <'
                    s += 'in={' + ', '.join(str(a) for a in self.nodes[j][v].incoming) + '}, '
                    s += 'out={' + ', '.join(str(a) for a in self.nodes[j][v].outgoing) + '}'
                    s += '>\n'
            s += '# (Outgoing) Arcs\n'
            s += '\n'.join(str(a) for a in self.alloutgoingarcs())
            if showIncoming:
                s += '\n# (Incoming) Arcs\n'
                s += '\n'.join(str(a) for a in self.allincomingarcs())
        else:
            # Short form
            s += '# Nodes\n'
            for (j, lyr) in enumerate(self.nodes):
                s += 'L' + str(j) + ': '
                s += ', '.join(str(v) for v in self.allnodes_in_layer(j)) + '\n'
            s += '# (Outgoing) Arcs\n'
            s += ', '.join(str(a) for a in self.alloutgoingarcs())
            if showIncoming:
                s += '\n# (Incoming) Arcs\n'
                s += ', '.join(str(a) for a in self.allincomingarcs())
        return s

    def __repr__(self):
        return 'MDD(' + repr(self.name) + ', ' + repr(self.nodes) + ')'

    def _get_node_info(self, node):
        """Get 'MDDNodeInfo' corresponding to 'node'.

        Get the 'MDDNodeInfo' object corresponding to the 'MDDNode'
        object 'node'. Note this function can *not* be used to populate
        the underlying dictionary; it can only be used to reference
        the object.
        """
        return self.nodes[node.layer][node]

    def _add_arc(self, newarc):
        """Add an arc to the MDD, without sanity checks."""
        self._get_node_info(newarc.tail).outgoing.add(newarc)
        self._get_node_info(newarc.head).incoming.add(newarc)

    def _remove_arc(self, rmvarc):
        """Remove an arc from the MDD, without sanity checks."""
        self._get_node_info(rmvarc.tail).outgoing.remove(rmvarc)
        self._get_node_info(rmvarc.head).incoming.remove(rmvarc)

    def _add_node(self, newnode):
        """Add a node to the MDD, without sanity checks."""
        # If an identical node already exists, its incoming and outgoing arcs will be ERASED!!!
        self.nodes[newnode.layer][newnode] = MDDNodeInfo()

    def _remove_node(self, rmvnode):
        """Remove a node from the MDD, without sanity checks."""
        for arc in self._get_node_info(rmvnode).incoming:
            self._get_node_info(arc.tail).outgoing.remove(arc)
        for arc in self._get_node_info(rmvnode).outgoing:
            self._get_node_info(arc.head).incoming.remove(arc)
        del self.nodes[rmvnode.layer][rmvnode]

    def _remove_nodes(self, rmvnodes):
        """Remove a list of nodes from the MDD, without sanity checks."""
        for v in rmvnodes:
            self._remove_node(v)

    def _append_new_layer(self):
        """Append a new layer to the MDD."""
        self.nodes.append(dict())

    def _clear(self):
        """Reset the MDD."""
        self.nodes = []

    def allnodes(self):
        """Return all MDDNodes in the MDD."""
        return chain.from_iterable(l.keys() for l in self.nodes)

    def allnodeitems_in_layer(self, layer):
        """Return all (MDDNode, MDDNodeInfo) pairs in a particular layer."""
        return self.nodes[layer].items()

    def allnodes_in_layer(self, layer):
        """Return all MDDNodes in a particular layer."""
        return self.nodes[layer].keys()

    def alloutgoingarcs(self):
        """Return all outgoing arcs in the MDD."""
        return chain.from_iterable(ui.outgoing for j in range(self.numArcLayers) for ui in self.nodes[j].values())

    def allincomingarcs(self):
        """Return all incoming arcs in the MDD."""
        return chain.from_iterable(ui.incoming for j in range(self.numArcLayers) for ui in self.nodes[j+1].values())

    def add_arc(self, newarc):
        """Add an arc to the MDD (with sanity checks).
        Args: newarc (MDDArc): arc to be added
        Raises:
            RuntimeError: head/tail node of arc does not exist
            ValueError: head and tail nodes must be one layer apart
        """
        if not newarc.tail in self.allnodes_in_layer(newarc.tail.layer):
            raise RuntimeError('tail node of arc does not exist')
        if not newarc.head in self.allnodes_in_layer(newarc.head.layer):
            raise RuntimeError('head node of arc does not exist')
        # if newarc.head.layer != newarc.tail.layer + 1:
        #     print(newarc)
        #     raise ValueError('head and tail must be one layer apart (%d != %d + 1)' % (newarc.head.layer, newarc.tail.layer))
        self._add_arc(newarc)

    def remove_arc(self, rmvarc):
        """Remove an arc from the MDD (with sanity checks).
        Args: rmvarc (MDDArc): arc to be removed
        Raises:
            RuntimeError: head/tail node of arc does not exist
            KeyError: no such incoming/outgoing arc exists in the MDD
        """
        if not rmvarc.tail in self.allnodes_in_layer(rmvarc.tail.layer):
            raise RuntimeError('tail node of arc does not exist')
        if not rmvarc.head in self.allnodes_in_layer(rmvarc.head.layer):
            raise RuntimeError('head node of arc does not exist')
        if not rmvarc in self._get_node_info(rmvarc.tail).outgoing:
            raise KeyError('cannot remove non-existent outgoing arc')
        if not rmvarc in self._get_node_info(rmvarc.head).incoming:
            raise KeyError('cannot remove non-existent incoming arc')
        self._remove_arc(rmvarc)

    def add_node(self, newnode):
        """Add a new node to the MDD (with sanity checks).
        Args: newnode (MDDNode): node to be added
        Raises:
            IndexError: the MDD does not contain the specified node layer
            ValueError: a duplicate node already exists in the MDD
        """
        if newnode.layer >= self.numNodeLayers or newnode.layer < 0:
            raise IndexError('node layer %d does not exist' % newnode.layer)
        if newnode in self.allnodes_in_layer(newnode.layer):
            raise ValueError('cannot add proposed node; duplicate node already exists')
        self._add_node(newnode)

    def remove_node(self, rmvnode):
        """Remove a node from the MDD (with sanity checks).

        Args:
            rmvnode (MDDNode): node to be removed

        Raises:
            IndexError: the MDD does not contain the specified node layer
            KeyError: no such node exists in the MDD
        """
        if rmvnode.layer >= self.numNodeLayers or rmvnode.layer < 0:
            raise IndexError('node layer %d does not exist' % rmvnode.layer)
        if not rmvnode in self.allnodes_in_layer(rmvnode.layer):
            raise KeyError('cannot remove non-existent node')
        self._remove_node(rmvnode)

    def find_neighbor_along_path(self, arc_names):
        cur_state = self.allnodes_in_layer(0)
        cur_state = list(cur_state)[0]
        neighbors_labels = []
        for j, name in enumerate(arc_names):
            next_state, neighbors = self._find_next_state(cur_state, name)
            cur_state = next_state
            neighbors_labels.append(neighbors)

        neighbors_labels.append(self.get_outgoing_of_state_at_layer(cur_state))
        return neighbors_labels

    def get_outgoing_of_state_at_layer(self, cur_state):
        return [x.label for x in self.nodes[cur_state.layer][cur_state].outgoing]

    def find_last_neighbor_along_path(self, arc_names):
        cur_state = self.allnodes_in_layer(0)
        cur_state = list(cur_state)[0]
        neighbors_labels = []
        for j, name in enumerate(arc_names):
            next_state, _ = self._find_next_state(cur_state, name)
            cur_state = next_state
            neighbors_labels.append([name])
        neighbors_labels.append(self.get_outgoing_of_state_at_layer(cur_state))
        return neighbors_labels

    def _find_next_state(self, cur_state, name):
        all_out_arcs = [x for x in self.nodes[cur_state.layer][cur_state].outgoing]
        neighbors = [x.label for x in all_out_arcs]
        for out_arc in all_out_arcs:
            if out_arc.label == name:
                return out_arc.head, neighbors

    # Default functions/args for GraphViz output
    @staticmethod
    def _default_ndf(state, layer):
        return 'label="%s"' % str(state)

    @staticmethod
    def _default_adf(label, layer):
        return 'label="%s"' % label

    _default_asa = {'key': lambda a: a.label}
    _default_nsa = {'key': lambda v: v.state, 'reverse': True}

    def output_to_dot(self, nodeDotFunc=None, arcDotFunc=None, arcSortArgs=None, nodeSortArgs=None, reverseDir=False, fname=None):
        """Write the graphical structure of the MDD to a file.
        Write the graphical structure of the MDD to a file (<MDDName>.gv) in
        the DOT language.  The MDD can then be visualized with GraphViz.
        Args:
            nodeDotFunc (Callable[[object, int], str]): nodeDotFunc(s,j)
                returns a string with the DOT options to use given node state
                's' in layer 'j'; if None (default), a sensible default is used
            arcDotFunc (Callable[[object, float, int], str]): arcDotFunc(l,w,j)
                returns a string with the DOT options to use given arc label
                'l', arc weight 'w', and tail node layer 'j'; if None (default),
                a sensible default is used
            arcSortArgs (dict): arguments specifying how to sort a list of arcs
                via list.sort() (i.e., 'key' and, optionally, 'reverse');
                GraphViz then attempts to order the arcs accordingly in the
                output graph; if arcSortArgs is None (default), no such order
                is enforced
            nodeSortArgs (dict): arguments specifying how to sort a list of
                nodes via list.sort() (i.e., 'key' and, optionally, 'reverse');
                GraphViz then attempts to order the nodes accordingly in the
                output graph; if nodeSortArgs is None (default), no such order
                is enforced
            reverseDir (bool): if True, show the MDD with arcs oriented in the
                opposite direction, so the terminal node appears at the top and
                the root node at the bottom (default: False)
            fname (str): name of output file; if None, default to <MDDName>.gv
        """

        # Use default output functions if unspecified
        if nodeDotFunc is None:
            nodeDotFunc = self._default_ndf
        if arcDotFunc is None:
            arcDotFunc = self._default_adf
        if reverseDir:
            iterRange = range(self.numArcLayers, 0, -1)
            (nextArcAttr, srcAttr, destAttr) = ('incoming', 'head', 'tail')
        else:
            iterRange = range(self.numArcLayers)
            (nextArcAttr, srcAttr, destAttr) = ('outgoing', 'tail', 'head')
        if fname is None:
            fname = '%s.gv' % self.name

        outf = open(fname, 'w')
        outf.write('digraph "%s" {\n' % self.name)
        outf.write('graph[fontname="Monospace Regular"];\nnode[fontname="Monospace Regular"];\nedge[fontname="Monospace Regular"];\n')
        if reverseDir:
            outf.write('edge [dir=back];\n')
        if arcSortArgs is not None:
            outf.write('ordering=out;\n')
        for v in self.allnodes():
            outf.write('%d[%s];\n' % (hash(v), nodeDotFunc(v.state, v.layer)))
        for j in iterRange:
            for (u, ui) in self.allnodeitems_in_layer(j):
                arcsinlayer = [a for a in getattr(ui, nextArcAttr)]
                if arcSortArgs is not None:
                    arcsinlayer.sort(**arcSortArgs)
                for arc in arcsinlayer:
                    outf.write('%d -> %d[%s];\n' % (hash(getattr(arc, srcAttr)), hash(getattr(arc, destAttr)), arcDotFunc(arc.label, arc.tail.layer)))
        if nodeSortArgs is not None:
            for j in range(self.numNodeLayers):
                nodesinlayer = [v for v in self.allnodes_in_layer(j)]
                if len(nodesinlayer) > 1:
                    nodesinlayer.sort(**nodeSortArgs)
                    for i in range(len(nodesinlayer) - 1):
                        outf.write('%d -> %d[style=invis];\n' % (hash(nodesinlayer[i]), hash(nodesinlayer[i+1])))
                    outf.write('{rank=same')
                    for v in nodesinlayer:
                        outf.write(';%d' % hash(v))
                    outf.write('}\n')
        outf.write('}')
        outf.close()

    def dumpJSON(self, fname=None, stateDumpFunc=repr, labelDumpFunc=repr):
        """Dump the MDD into a JSON file.
        Args:
            fname (str): name of json file (default: self.name + '.json')
            stateDumpFunc (Callable[[object], str]): stateDumpFunc(s) returns
                a string representation of the node state 's' (default: repr)
            labelDumpFunc (Callable[[object], str]): labelDumpFunc(l) returns
                a string representation of the arc label 'l' (default: repr)
        """
        if fname is None:
            fname = self.name + '.json'
        dataList = []
        dataList.append({'Type': 'name', 'name': self.name})
        for v in self.allnodes():
            dataList.append({'Type': 'node', 'layer': v.layer, 'state': stateDumpFunc(v.state), 'id': hash(v)})
        for a in self.alloutgoingarcs():
            dataList.append({'Type': 'arc', 'label': labelDumpFunc(a.label), 'tail': hash(a.tail), 'head': hash(a.head)})
        outf = open(fname, 'w')
        dump(dataList, outf)
        outf.close()

    def relax_mdd(self, maxWidth):
        """Load an MDD from a JSON file."""
        j = 1
        node_idx = 40
        print("numArcLayers: {}".format(self.numArcLayers))
        while j < self.numArcLayers and len(self.allnodes_in_layer(j)) < maxWidth:
            print(j, len(self.allnodes_in_layer(j)), maxWidth)
            nodesinlayer = [v for v in self.allnodes_in_layer(j)]
            for v in nodesinlayer:
                length = len(self.nodes[j][v].incoming)
                print("layer {} incoming edges: {}".format(j, length))
                if length <= 1:
                    continue
                rand_bits = np.random.choice(length, 1)
                if len(self.nodes[j]) > maxWidth - 1:
                    print("node canot be splitted, beacause reaching the bound!:", len(self.nodes[j]))
                    continue
                # node split
                new_nodes1 = MDDNode(layer=j, state="u"+str(node_idx))
                new_nodes1_income = set()
                new_nodes2 = MDDNode(layer=j, state="u"+str(node_idx+1))
                new_nodes2_income = set()

                self.add_node(new_nodes1)
                self.add_node(new_nodes2)
                node_idx += 2
                for i, a in enumerate(self.nodes[j][v].incoming):
                    if i == rand_bits:
                        newarc = MDDArc(a.label, a.tail, new_nodes1)
                        self.add_arc(newarc)
                        new_nodes1_income.add(a.label)
                    else:
                        newarc = MDDArc(a.label, a.tail, new_nodes2)
                        self.add_arc(newarc)
                        new_nodes2_income.add(a.label)
                print("split the incoming edges:", len(self.nodes[j][v].incoming), len(self.nodes[j][new_nodes1].incoming), len(self.nodes[j][new_nodes2].incoming))
                # arc filtering
                for x in self.nodes[j][v].outgoing:
                    if not x.label in new_nodes1_income:
                        newarc1 = MDDArc(x.label, new_nodes1, x.head)
                        self.add_arc(newarc1)
                for x in self.nodes[j][v].outgoing:
                    newarc2 = MDDArc(x.label, new_nodes2, x.head)
                    self.add_arc(newarc2)
                print("split the outgoings edges:", len(self.nodes[j][v].outgoing),
                      len(self.nodes[j][new_nodes1].outgoing),
                      len(self.nodes[j][new_nodes2].outgoing),
                      len(self.nodes[j][new_nodes1].outgoing) + len(self.nodes[j][new_nodes2].outgoing))
                print("add node: {} {}, remove node: {}".format(new_nodes1, new_nodes2, v))
                self.remove_node(v)
            j += 1
            if j >= self.numArcLayers:
                j = 1

    def loadJSON(self, json_content):
        """Load an MDD from a JSON file."""
        self._clear()
        dataList = json_content
        nodeDict = dict()
        self.name = dataList['mdd_name']
        for item in dataList['init']:
            if item['Type'] == 'node':
                while int(item['layer']) >= self.numNodeLayers:
                    self._append_new_layer()
                newnode = MDDNode(int(item['layer']), item['state'])
                self.add_node(newnode)
                nodeDict[item['id']] = newnode
            elif item['Type'] == 'arc':
                newarc = MDDArc(item['label'], nodeDict[item['tail']], nodeDict[item['head']])
                self.add_arc(newarc)
            else:
                raise ValueError('Unknown item type: check input file format')



