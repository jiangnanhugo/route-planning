#!/usr/bin/zsh

for loc in 10 12 14 16 18 20 22 24 26 28
do
	python example_mdd_tsp.py --max_loc $loc > relaxed_mdd_$loc.log &
done