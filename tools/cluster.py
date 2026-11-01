import torch
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN


def clustering_dict(n_clusters, data_dict, algorithm='auto'):
    pass
    keys = list(data_dict.keys())
    tensors = [data_dict[k] for k in keys]
    

    data = []
    for tensor in tensors:

        if tensor.is_cuda:
            arr = tensor.cpu().detach().numpy().flatten()
        else:
            arr = tensor.detach().numpy().flatten()
        data.append(arr)
    

    data_matrix = np.vstack(data)
    n_samples, n_features = data_matrix.shape
    

    if algorithm == 'auto':
        if n_samples > 10000:
            algorithm = 'minibatch'
        elif n_features > 100:
            algorithm = 'hdbscan'
        else:
            algorithm = 'kmeans'
    

    if algorithm == 'kmeans':

        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(data_matrix)
        labels = kmeans.labels_
    
    elif algorithm == 'minibatch':
        mbk = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, 
                             n_init='auto', batch_size=1024)
        mbk.fit(data_matrix)
        labels = mbk.labels_
    
    elif algorithm == 'dbscan':
        db = DBSCAN(eps=0.5, min_samples=5)
        db.fit(data_matrix)
        labels = db.labels_
        unique_labels = set(labels)
        n_found_clusters = len(unique_labels) - (-1 in unique_labels)
        
        if n_found_clusters < n_clusters:
            kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
            kmeans.fit(data_matrix)
            labels = kmeans.labels_
    else:
        raise ValueError(f"未知算法: {algorithm}")
    

    cluster_labels = {}
    cluster_members = {}
    

    for i, key in enumerate(keys):
        cluster_labels[key] = int(labels[i])
        

        label = int(labels[i])
        if label not in cluster_members:
            cluster_members[label] = []
        cluster_members[label].append(key)
    
    return cluster_labels, cluster_members

def get_cluster_average(cluster_members, cat_grad_dict):
    cluster_grad={}
    for cluster, members in cluster_members.items():
        print(cluster, members)
        cluster_grad[cluster] = sum(cat_grad_dict[member] for member in members) / len(members)
    return cluster_grad