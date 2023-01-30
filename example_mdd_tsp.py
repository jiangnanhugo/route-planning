from mdd.utils import get_mdd
from pympler import asizeof
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MDD.")

    parser.add_argument("--max_loc", required=True, type=int, help='The maximum loccations')
    args = parser.parse_args()

    for i in range(20):
        graph = get_mdd(args.max_loc, args.max_loc, 2 ** i)
        print("==========", asizeof.asizeof(graph) / (1024 * 1024))
    # print(graph.__str__(showLong=True))
