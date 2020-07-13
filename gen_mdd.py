import pickle, sys

import mdd


##### Main Program #####
def main():
    prob_file = "output/4city_exp1.prob"
    max_width = 100

    # read problem
    inp = open(prob_file, 'rb')
    prob = pickle.load(inp)
    inp.close()
    prob.print_prob(sys.stdout)

    # build mdd
    mdd0 = mdd.MDD_TSP(prob.paired_dist, prob.startp, prob.endp,
                       prob.max_duration, prob.max_stops,
                       max_width)

    mdd0.filter_refine_preparation()

    mdd0.filter_refine()

    mdd0.add_last_node_forever()

    # print mdd
    mdd0.print_mdd(sys.stdout)


if __name__ == '__main__':
    main()



