#!/usr/bin/python

import sys, random, copy, pickle
import mdd
import data_utils


def floyd(paired_dist):
    path = copy.copy(paired_dist)
    n = len(paired_dist)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if path[i][j] > path[i][k] + path[k][j]:
                    path[i][j] = path[i][k] + path[k][j]
    return path


def read_tsp_file_full_matrix(tsp_file):
    inp = open(tsp_file, 'r')
    mtx = []
    start_reading = False
    lines_read = 0
    for l in inp.readlines():
        if l.startswith('DIMENSION'):
            tt = l[:-1].split()
            dim = int(tt[1])
        elif l.startswith('EDGE_WEIGHT_SECTION'):
            start_reading = True
        elif start_reading and lines_read < dim:
            tt = l[:-1].split()
            line = [float(itm) for itm in tt]
            mtx.append(line)
            lines_read += 1
    inp.close()
    return mtx

####

tsp_file = "./benchmarks/TSPLIB95/bays29.tsp"
mdd_file = "output/bays29_s0_e0_md1000_ms10_w100.mdd"

pairwise_dist = read_tsp_file_full_matrix(tsp_file)

[print(line) for line in pairwise_dist]

paired_shortest_path = floyd(pairwise_dist)

startp = 0
endp = 0

max_duration = 1000000
max_stops = 29
max_width = 100

##
# mdd = mdd.MDD_TSP(paired_shortest_path, startp, endp, max_duration, max_stops, max_width)

# mdd.filter_refine_preparation()

# mdd.print_mdd(sys.stdout)
# mdd.filter_refine()

# print('===== after filter and refining =====')
# oup = open(mdd_file, 'w')
# mdd.print_mdd(oup) # sys.stdout
# oup.close()

