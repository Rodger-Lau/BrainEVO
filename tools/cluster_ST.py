import torch
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
def clustering_2d_list(n_clusters, tensor_2d_list, algorithm='auto', eps=0.5, min_samples=5):
    pass
    if not isinstance(tensor_2d_list, list) or not all(isinstance(row, list) for row in tensor_2d_list):
        raise ValueError("输入必须是二维列表")
    

    positions = []
    tensors = []

    for row_idx, row in enumerate(tensor_2d_list):
        for col_idx, tensor in enumerate(row):
            positions.append((row_idx, col_idx))
            tensors.append(tensor)
    

    if len(tensors) == 0:
        return {}
    

    data = []
    for tensor in tensors:
        if tensor.is_cuda:
            arr = tensor.cpu().detach().numpy().flatten()
        else:
            arr = tensor.detach().numpy().flatten()
        data.append(arr)
    
    data_matrix = np.vstack(data)

    scaler = StandardScaler()
    data_matrix_scaled = scaler.fit_transform(data_matrix)

    if algorithm == 'auto':
        if len(tensors) > 10000:
            algorithm = 'minibatch'
        else:
            algorithm = 'kmeans'
    

    if algorithm == 'kmeans':
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
        kmeans.fit(data_matrix)
        labels = kmeans.labels_
    
    elif algorithm == 'minibatch':
        mbk = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, 
                             n_init='auto', batch_size=1024)
        mbk.fit(data_matrix)
        labels = mbk.labels_
    elif algorithm == 'dbscan':

        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan.fit(data_matrix_scaled)
        labels = dbscan.labels_
    
    else:
        raise ValueError(f"不支持算法: {algorithm}")
    

    cluster_positions = {}
    
    for pos, label in zip(positions, labels):

        label = int(label)
        if label not in cluster_positions:
            cluster_positions[label] = []
        cluster_positions[label].append(pos)
    
    return cluster_positions

def get_cluster_average(cluster_members, cat_grad_dict):
    cluster_grad={}
    for cluster, members in cluster_members.items():
        print(cluster, members)
        cluster_grad[cluster] = sum(cat_grad_dict[member[0]][member[1]] for member in members) / len(members)
    return cluster_grad