from .core import Langraph


def main() -> None:
    graph = Langraph()
    print(graph.describe())


if __name__ == "__main__":
    main()
