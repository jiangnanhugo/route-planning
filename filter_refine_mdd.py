import copy
from itertools import product
import sys
from mdd import MDD_TSP


def floyd_warshall(paired_dist):
    path = copy.copy(paired_dist)
    n = len(paired_dist)
    rn = range(n)
    for k, i, j in product(rn, repeat=3):
        path_ik_to_kj = path[i][k] + path[k][j]
        if path[i][j] > path_ik_to_kj:
            path[i][j] = path_ik_to_kj
    return path


def read_tsp_file_full_matrix(tsp_file):
    inp = open(tsp_file, 'r')
    mtx = []
    start_reading = False
    lines_read = 0
    dim = -1
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


def main():
    tsp_file = "./benchmarks/TSPLIB95/bays29.tsp"
    mdd_file = "output/bays29_s0_e0_md1000_ms10_w100.mdd"

    pairwise_dist = read_tsp_file_full_matrix(tsp_file)

    [print(line) for line in pairwise_dist]

    paired_shortest_path = floyd_warshall(pairwise_dist)

    startp = 0
    endp = 0

    max_duration = 1000000
    max_stops = 29
    max_width = 100

    mdd = MDD_TSP(paired_shortest_path, startp, endp, max_duration, max_stops, max_width)

    mdd.filter_refine_preparation()

    mdd.print_mdd(sys.stdout)
    mdd.relax_mdd()

    print('===== after filter and refining =====')
    oup = open(mdd_file, 'w')
    mdd.print_mdd(oup)  # sys.stdout
    oup.close()


if __name__ == '__main__':
    main()



