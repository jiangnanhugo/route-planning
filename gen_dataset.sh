for loc in $(seq 27 1 29)
do
	python3 gen_synthetic_schedules.py --max_stops $loc  --max_duration 100000
done
