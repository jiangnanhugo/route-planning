for loc in $(seq 2 1 29)
do
	echo $loc
	echo "with crisp"
	cat dataset/"$loc"locations/*crisp1.out|grep "valid routes="|sort|tail -n 1
	echo "without crisp"
	cat dataset/"$loc"locations/*crisp0.out|grep "valid routes=="|sort|tail -n 1
done


