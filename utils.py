import numpy as np
import pandas as pd

from sklearn.metrics import average_precision_score, roc_auc_score, ndcg_score


def load_network(dataset):
    """load network"""
    network = pd.read_csv('../data/%s/network.csv' % dataset, header=None)
    network.columns = ['user', 'item', 'rating']
    network[['user', 'item']] = network[['user', 'item']].astype(str)
    network = network.values
    
    # scale rating to the interval between -1 and 1
    min_rating = np.min(network[:, 2])
    max_rating = np.max(network[:, 2])
    network[:, 2] = (network[:, 2] - (min_rating + max_rating) / 2) / (max_rating - min_rating) * 2
    
    print('After scaling rating, the min and max rating:', np.min(network[:, 2]), np.max(network[:, 2]))
    
    return network


def load_gt(dataset):
    """load ground truth"""
    gt = pd.read_csv('../data/%s/gt.csv' % dataset, header=None)
    gt.columns = ['user', 'label']
    gt['user'] = gt['user'].astype(str)
    gt = gt.values

    normals = set([user for user, label in gt if label == 0])
    frauds = set([user for user, label in gt if label == 1])
    
    print('# normals:', len(normals))
    print('# frauds:', len(frauds))
    return normals, frauds


def evaluate(y_pred, y_true):
    """
    record the result to disk for further visualization
    display numeric evaluation results
    """
    
    # average precision
    overall_ap = average_precision_score(y_true, y_pred)
    ap_200 = average_precision_score(np.concatenate([y_true[:100], y_true[-100:]]),
                                     np.concatenate([y_pred[:100], y_pred[-100:]]))
    print('AP: {:.2f}%'.format(overall_ap * 100),
          'AP@200: {:.2f}%'.format(ap_200 * 100))
    
    # auc
    overall_auc = roc_auc_score(y_true, y_pred)
    print('AUC: {:.2f}%'.format(overall_auc * 100))
    
    # precision@k
    precision_50 = int(np.sum(y_true[:50]))
    precision_100 = int(np.sum(y_true[:100]))
    print('Precision@50: {:d}%'.format(precision_50),
          'Precision@100: {:d}%'.format(precision_100))
    
    # Precision values @ 20% Recall & 80% Recall
    precision_recall_20 = 0
    precision_recall_80 = 0
    total_frauds = np.sum(y_true)
    cur_frauds = 0
    cur_samples = 0
    for i in range(len(y_true)):
        cur_samples += 1
        if y_true[i] == 1:
            cur_frauds += 1
        
        if cur_frauds == int(total_frauds * 0.2):
            precision_recall_20 = cur_frauds / cur_samples
        if cur_frauds == int(total_frauds * 0.8):
            precision_recall_80 = cur_frauds / cur_samples
    print('Precision@20%Recall: {:.2f}%'.format(precision_recall_20 * 100))
    print('Precision@80%Recall: {:.2f}%'.format(precision_recall_80 * 100))
    
    
    # NDCG
    ndcg_200 = ndcg_score(np.concatenate([y_true[:100], y_true[-100:]]).reshape(1, -1),
                          np.concatenate([y_pred[:100], y_pred[-100:]]).reshape(1, -1))
    print('NDCG@200: {:.2f}%'.format(ndcg_200 * 100))
    