for loc in $(seq 2 1 29)
do
	echo $loc
	echo "w crisp"
	cat dataset/"$loc"locations/*mask1.out|grep "valid: gen="|sort|tail -n 1
	echo "w/o crisp"
	cat dataset/"$loc"locations/*mask0.out|grep "valid: gen="|sort|tail -n 1
done

