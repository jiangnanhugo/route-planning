set -x
loc=$1
use_crisp=$2
post_proces_type1=$3
post_proces_type2=$4
cuda=$5
folder=dataset/${loc}locations

max_width=$6
batchnum=100
batchnum_summary=10

d_n_hidden=100

g_n_hidden=100
g_n_input=20

lr=0.001
glr=0.001


data_file=$folder/bays29_s0_e0_md1000_ms${loc}_random_reward_ne10000_w1000_gt1000.data
prob_file=$folder/bays29_s0_e0_md1000_ms${loc}_random_reward.prob


echo "$prob_file use_crisp=$use_crisp post_proces_type1=$post_proces_type1, post_proces_type2=$post_proces_type2, maxwidth=$max_width"
python3 gan_with_crisp.py --data_file $data_file --prob_file $prob_file --max_width $max_width --batch_size $batchnum \
--test_size $batchnum_summary  --d_n_hidden $d_n_hidden --g_n_hidden $g_n_hidden --g_n_input $g_n_input \
--lr $lr --glr $glr --cuda $cuda --use_crisp $use_crisp --use_post_process_type2 $post_proces_type1 \
--use_post_process_type2 $post_proces_type2 > $data_file.crisp${use_crisp}.width${max_width}.post_proces_type${post_proces_type1}.post_proces_type${post_proces_type2}.out

