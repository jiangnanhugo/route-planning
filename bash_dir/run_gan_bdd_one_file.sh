!/bin/sh

folder=output

# data_file=$folder/bays29_s0_e0_md1000_ms10_random_reward_ne10000_w1000_gt1000.data
# prob_file=$folder/bays29_s0_e0_md1000_ms10_random_reward.prob

# data_file=$folder/bays29_s0_e0_md1000_ms6_random_reward_ne100000_w1000.data
# prob_file=$folder/bays29_s0_e0_md1000_ms6_random_reward.prob

batchnum=100 #00
batchnum_summary=1000

d_n_hidden=100

g_n_hidden=100
g_n_input=20

lr=0.01
glr=0.01

cuda=0

#0
data_file=$folder/6city_exp1.data
prob_file=$folder/6city_exp1.prob

use_mask=0

echo "$prob_file mask=$use_mask"
python gan_bdd.py --data_file $data_file --prob_file $prob_file --batchnum $batchnum --batchnum_summary $batchnum_summary  --d_n_hidden $d_n_hidden --g_n_hidden $g_n_hidden --g_n_input $g_n_input --lr $lr --glr $glr --cuda $cuda --mask $use_mask > $data_file.mask${use_mask}.out

use_mask=1

echo "$prob_file mask=$use_mask"
python gan_bdd.py --data_file $data_file --prob_file $prob_file --batchnum $batchnum --batchnum_summary $batchnum_summary  --d_n_hidden $d_n_hidden --g_n_hidden $g_n_hidden --g_n_input $g_n_input --lr $lr --glr $glr --cuda $cuda --mask $use_mask > $data_file.mask${use_mask}.out


# #1
# data_file=$folder/bays29_s0_e0_md1000_ms12_random_reward_ne10000_w1000_gt1000.data
# prob_file=$folder/bays29_s0_e0_md1000_ms12_random_reward.prob

# use_mask=0

# echo "$prob_file mask=$use_mask"
# python gan_bdd.py --data_file $data_file --prob_file $prob_file --batchnum $batchnum --batchnum_summary $batchnum_summary  --d_n_hidden $d_n_hidden --g_n_hidden $g_n_hidden --g_n_input $g_n_input --lr $lr --glr $glr --cuda $cuda --mask $use_mask > $data_file.mask${use_mask}.out

# use_mask=1

# echo "$prob_file mask=$use_mask"
# python gan_bdd.py --data_file $data_file --prob_file $prob_file --batchnum $batchnum --batchnum_summary $batchnum_summary  --d_n_hidden $d_n_hidden --g_n_hidden $g_n_hidden --g_n_input $g_n_input --lr $lr --glr $glr --cuda $cuda --mask $use_mask > $data_file.mask${use_mask}.out


# #2
# data_file=$folder/bays29_s0_e0_md1000_ms6_random_reward_ne100000_w1000.data
# prob_file=$folder/bays29_s0_e0_md1000_ms6_random_reward.prob

# use_mask=0

# echo "$prob_file mask=$use_mask"
# python gan_bdd.py --data_file $data_file --prob_file $prob_file --batchnum $batchnum --batchnum_summary $batchnum_summary  --d_n_hidden $d_n_hidden --g_n_hidden $g_n_hidden --g_n_input $g_n_input --lr $lr --glr $glr --cuda $cuda --mask $use_mask > $data_file.mask${use_mask}.new.out

# use_mask=1

# echo "$prob_file mask=$use_mask"
# python gan_bdd.py --data_file $data_file --prob_file $prob_file --batchnum $batchnum --batchnum_summary $batchnum_summary  --d_n_hidden $d_n_hidden --g_n_hidden $g_n_hidden --g_n_input $g_n_input --lr $lr --glr $glr --cuda $cuda --mask $use_mask > $data_file.mask${use_mask}.new.out

echo "complete!"
