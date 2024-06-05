from graph import Graph
from utils import *


if __name__ == '__main__':
    datasets = ['finefoods']

    for dataset in datasets:
        G = Graph(dataset=dataset)
        
        probs = G.FD_SpaN()
        evaluate(probs[:, 0], probs[:, 1])
