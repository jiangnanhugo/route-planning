set -x
for width in 16 32
do
	n_loc=5
	use_crisp=True
	post_proces_type1=False
	post_proces_type2=False
	cuda_device=0
    ./bash_dir/run_with_loc.sh $n_loc $use_crisp $post_proces_type1 $post_proces_type2 $cuda_device $width
done

