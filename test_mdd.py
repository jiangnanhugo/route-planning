import sys, copy, pickle

import mdd, data_utils

#from gen_synthetic_schedules import floyd

def floyd(paired_dist):
    path = copy.copy(paired_dist)
    n = len(paired_dist)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if path[i][j] > path[i][k] + path[k][j]:
                    path[i][j] = path[i][k] + path[k][j]
    return path

##### Main Program #####

# paired_dist = [[0.0, 1.0, 3.0],\
#                [1.0, 0.0, 5.0],\
#                [3.0, 5.0, 0.0]]

# startp = 0
# endp = 0

# max_duration = 6.0
# max_stops = 2
# max_width = 100

# mdd = mdd.MDD_TSP(paired_dist, startp, endp, max_duration, max_stops, max_width)
# #mdd.print_mdd(sys.stdout)

# mdd.earliest_time()
# mdd.latest_time()

# mdd.all_down()
# mdd.some_down()
# mdd.all_up()
# mdd.some_up()

# #mdd.print_mdd('test.txt')
# mdd.print_mdd(sys.stdout)

# mdd.filter_refine()

#### Input 2:
## mdd stuff
paired_dist = [[0.0, 1.0, 10.0, 15.0],\
               [1.0, 0.0, 2.0, 12.0],\
               [10.0, 2.0, 0.0, 3.0],\
               [15.0, 12.0, 3.0, 0.0]]

paired_shortest_path = floyd(paired_dist)
print('paired_shortest_path', paired_shortest_path)

startp = 0
endp = 0

max_duration = 20.0
max_stops = 3
max_width = 100

## reward generator 
rewards = [[0., 5., 1., 1.],\
           [0., 2., 7., 2.],\
           [0., 3., 3., 8.],\
           [0., 0., 0., 0.]]
# ,\
#            [0., 1., 1., 1.],\
#            [0., 1., 1., 1.],\
#            [0., 1., 1., 1.]]

data_file = "output/4city_exp1.data"
data_gen = data_utils.ScheduleDataGen(data_file, max_stops, \
                                      len(paired_dist))
next_data, next_visit = data_gen.next_data(3)

print('next_data=', next_data)
print('next_visit=', next_visit)


output_prob_instance = "output/4city_exp1.prob"
prob = data_utils.ScheduleProb()
prob.init_by_assign(paired_shortest_path, startp, endp, \
                    max_duration, max_stops, rewards)
oup = open(output_prob_instance, 'wb')
pickle.dump(prob, oup)
oup.close()
