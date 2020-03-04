for loc in $(seq 2 1 29)
do
	echo $loc
	echo "w crisp"
	cat dataset/"$loc"locations/*mask1.out|grep "norm" > dataset/"$loc"locations/normalized_reward_mask1.out
	echo "w/o crisp"
	cat dataset/"$loc"locations/*mask0.out|grep "norm" > dataset/"$loc"locations/normalized_reward_mask0.out
done
