from mdd.utils import get_mdd


if __name__ == '__main__':
    graph = get_mdd(5, 3, 200)
    print(graph)
    # print(graph.__str__(showLong=True))
