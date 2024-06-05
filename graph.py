import time
import numpy as np
import matplotlib.pyplot as plt

from utils import *
from collections import defaultdict


class Graph:

    def __init__(self, dataset='finefoods'):
        self.dataset = dataset
        print('Handling with dataset:', self.dataset)
        
        # load data from csv files
        self.adj = load_network(dataset=dataset)

        # different training percentage
        # np.random.shuffle(self.adj)
        # self.adj = self.adj[:int(len(self.adj) * p), :]

        # unique users and items
        self.users = np.unique(self.adj[:, 0])
        self.items = np.unique(self.adj[:, 1])

        # normal and anomalous users
        self.normals, self.frauds = load_gt(dataset=dataset)

        # in- and out-edges
        self.in_edges = defaultdict(list)
        self.out_edges = defaultdict(list)
        for i, (user, item, _) in enumerate(self.adj):
            self.out_edges[user].append(i)
            self.in_edges[item].append(i)

    def _fairness_to_probs(self, fairness, method):
        probs = []
        for user, v in fairness.items():
            if user in self.normals:
                probs.append([1 - v, 0])
            if user in self.frauds:
                probs.append([1 - v, 1])
        probs = np.array(sorted(probs, key=lambda x: x[0], reverse=True))

        # save results to disk
        # np.savetxt('../experiments/%s_%s_result.txt' % (self.dataset, method), probs, fmt=['%f', '%d'], delimiter=' ')
        return probs
    
    def FD_SpaN(self, epochs=50, q=0.0, t=0.5, h=0.5, epsilon=1e-3, tau=5):
        if self.dataset == 'finefoods':
            tau = 5
        elif self.dataset == 'instruments':
            tau = 7
            
        # initialization
        quality = defaultdict(lambda: q)
        trust = np.ones(len(self.adj)) * t
        honesty = defaultdict(lambda: 0.)
        
        # iterations
        start_time = time.time()
        for epoch in range(epochs):
            # init
            start = time.time()
            cost_q, cost_t, cost_h = 0, 0, 0
            
            # update quality for each item
            for item in self.items:
                indexes = self.in_edges[item]
                update_q = (np.dot(self.adj[indexes, 2], trust[indexes]) + np.sum(1 - trust[indexes]) * q) / len(indexes)
                
                cost_q += abs(update_q - quality[item])
                quality[item] = update_q
            
            # update default value
            q = np.mean(list(quality.values()))
            cost_q = cost_q / len(self.items)
            
            # update trust score for each rating
            for i, (user, item, rating) in enumerate(self.adj):
                sig = np.mean(trust[self.in_edges[item]])
                gamma1 = np.tanh(len(self.out_edges[user]) / tau)
                
                dev = abs(rating - quality[item]) / 2
                update_t = (sig * (1 - dev) + gamma1 * honesty[user] + (1 - sig) * t) / (1 + gamma1)
 
                cost_t += abs(update_t - trust[i])
                trust[i] = update_t
                
            # update default values
            cost_t = cost_t / len(self.adj)
            t = np.mean(trust)
            
            # update honesty for each user
            for user in self.users:
                # parameters
                indexes = self.out_edges[user]
                update_h = (np.sum(trust[indexes]) + h) / (len(indexes) + 1)
                cost_h += abs(update_h - honesty[user])
                honesty[user] = update_h
            
            # update default value
            h = np.mean(list(honesty.values()))
            cost_h = cost_h / len(self.users)
            
            # display training info
            print('epoch {:03d} =>'.format(epoch),
                  'error_q: {:.4f};'.format(cost_q),
                  'error_t: {:.4f};'.format(cost_t),
                  'error_h: {:.4f};'.format(cost_h),
                  'time cost: {:.2f}s;'.format(time.time() - start))
            
            print('epoch %d => time cost: %.2fs; %.4f %.4f' % (epoch, time.time() - start, cost_q, cost_t))
            if (cost_q < epsilon) and (cost_t < epsilon) and (cost_h < epsilon):
                print('early stop.')
                break
        
        print('total time cost: %.2fs' % (time.time() - start_time))
        
        # convert to fraud probabilities
        return self._fairness_to_probs(honesty, method='x')

