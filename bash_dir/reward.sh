for loc in $(seq 2 1 29)
do
	echo $loc
	echo "w crisp"
	cat dataset/"$loc"locations/*crisp1.out|grep "norm-reward" > dataset/"$loc"locations/normalized_reward_crisp1.out
	echo "w/o crisp"
	cat dataset/"$loc"locations/*crisp0.out|grep "norm" > dataset/"$loc"locations/normalized_reward_crisp0.out
done
