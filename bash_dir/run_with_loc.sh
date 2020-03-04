loc=$1
use_mask=$2
cuda=$3
folder=dataset/"$loc"locations

max_width=1000
batchnum=100 #00
batchnum_summary=1000

d_n_hidden=100

g_n_hidden=100
g_n_input=20

lr=0.001
glr=0.001


data_file=$folder/bays29_s0_e0_md1000_ms"$loc"_random_reward_ne10000_w1000_gt1000.data
prob_file=$folder/bays29_s0_e0_md1000_ms"$loc"_random_reward.prob


echo "$prob_file mask=$use_mask"
python gan_bdd.py --data_file $data_file --prob_file $prob_file --max_width $max_width --batchnum $batchnum --batchnum_summary $batchnum_summary  --d_n_hidden $d_n_hidden --g_n_hidden $g_n_hidden --g_n_input $g_n_input --lr $lr --glr $glr --cuda $cuda --mask $use_mask > $data_file.mask${use_mask}.out
