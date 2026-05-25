import pickle
import random
import numpy as np
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
from tqdm import tqdm
def min_euclidean_distance(query_vector, vector_store):
    n_i, n_j, d = vector_store.shape
    reshaped_store = vector_store.reshape(-1, d)
    differences = (reshaped_store - query_vector) / (query_vector.shape[0] * query_vector.shape[1])
    distances = np.linalg.norm(differences, axis=1)
    distances = distances.reshape(n_i, n_j)
    min_index_flat = np.argmin(distances)
    min_i = min_index_flat // n_j
    min_j = min_index_flat % n_j
    min_distance = distances[min_i, min_j]
    
    return min_distance
def get_new_re(new_train_loader,personal_extractor):
    personal_extractor.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    personal_extractor.to(device)
    total_features = None
    total_samples = 0
    with torch.no_grad():
        for batch in new_train_loader:
            inputs = batch[0].to(device)
            features = personal_extractor(inputs)
            if total_features is None:
                total_features = torch.zeros(features.shape[1], device=device)
            total_features += features.sum(dim=0)
            total_samples += features.shape[0]
    if total_samples > 0:
        mean_feature = total_features / total_samples
        mean_feature_np = mean_feature.cpu().numpy()
        print(f"Calculated mean feature vector for {total_samples} samples")
        print("Mean feature shape:", mean_feature_np.shape)
    else:
        mean_feature_np = np.array([])
        print("No samples processed")
    return mean_feature_np
def get_G_store(personal_extractor,train_loaders):
    personal_extractor.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    personal_extractor.to(device)

    num_i = len(train_loaders)
    num_j = len(train_loaders[0]) if num_i > 0 else 0
    sample_data = next(iter(train_loaders[0][0]))[0].to(device)
    with torch.no_grad():
        sample_feature = personal_extractor(sample_data)
    feature_dim = sample_feature.shape[-1]
    G_store = np.zeros((num_i, num_j, feature_dim))
    for i in range(num_i):
        for j in range(num_j):
            loader = train_loaders[i][j]
            total_features = torch.zeros(feature_dim).to(device)
            total_samples = 0
            for batch in tqdm(loader, desc=f"Processing domain ({i},{j})"):
                data = batch[0].to(device)         
                with torch.no_grad():
                    features = personal_extractor(data)
                total_features += features.sum(dim=0)
                total_samples += features.shape[0]
            if total_samples > 0:
                domain_mean = (total_features / total_samples).cpu().numpy()
                G_store[i, j] = domain_mean
    return G_store
class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()

class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean



def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()

def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def load_adj(pkl_filename, adjtype):
    #sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    adj_mx = np.load(pkl_filename)
    if adjtype == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adjtype == "normlap":
        adj = [calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adjtype == "symnadj":
        adj = [sym_adj(adj_mx)]
    elif adjtype == "transition":
        adj = [asym_adj(adj_mx)]
    elif adjtype == "doubletransition":
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    elif adjtype == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    else:
        error = 0
        assert error, "adj type not defined"
    #return sensor_ids, sensor_id_to_ind, adj
    return adj


def load_dataset(dataset_dir, batch_size, valid_batch_size= None, test_batch_size=None):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scaler
    return data

def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred,real,0.0).item()
    mape = masked_mape(pred,real,0.0).item()
    rmse = masked_rmse(pred,real,0.0).item()
    return mae,mape,rmse

def choose_sublist(sorted_gradients_list, ratio):
    if not 0 < ratio <= 1:
        raise ValueError("ratio必须在(0,1]之间")

    n = len(sorted_gradients_list)
    k = max(1, int(n * ratio))  
    mask = set(random.sample(range(n), k))  # 随机索引集合
    return [x for i, x in enumerate(sorted_gradients_list) if i in mask]

import numpy as np
from itertools import permutations

def find_path(cluster_grad):
    """
    找到一条穿过所有cluster向量的最短路径（开放路径，起点为范数最小的向量）
    
    参数:
        cluster_grad: dict, {cluster_id: vector}
                      例如 {0: [0,1], 1: [1,0], 2: [1,1]}
    返回:
        list, 有序的cluster编号列表，表示最短访问顺序（起点为范数最小的cluster）
    """
    cluster_ids = list(cluster_grad.keys())
    n = len(cluster_ids)
    
    if n <= 1:
        return cluster_ids
    
    # 将向量转为numpy数组
    vectors = {cid: np.array(cluster_grad[cid], dtype=float) for cid in cluster_ids}
    
    # 预计算距离矩阵
    dist_matrix = {}
    for i in cluster_ids:
        for j in cluster_ids:
            if i != j:
                dist_matrix[(i, j)] = np.linalg.norm(vectors[i] - vectors[j])
    
    # ★ 找到范数最小的cluster，固定为起点
    start = min(cluster_ids, key=lambda cid: np.linalg.norm(vectors[cid]))
    
    # 剩余需要排列的节点
    remaining = [cid for cid in cluster_ids if cid != start]
    
    if n <= 11:
        # -------- 精确解：起点固定，只枚举剩余 (n-1)! 种排列 --------
        return _find_path_exact(start, remaining, dist_matrix)
    else:
        # -------- 启发式解：最近邻贪心（强制从start出发） --------
        return _find_path_nearest_neighbor(start, cluster_ids, dist_matrix)


def _path_total_distance(path, dist_matrix):
    """计算一条路径的总距离"""
    total = 0.0
    for i in range(len(path) - 1):
        total += dist_matrix[(path[i], path[i+1])]
    return total


def _find_path_exact(start, remaining, dist_matrix):
    """起点固定，暴力枚举剩余节点的排列"""
    best_path = None
    best_dist = float('inf')
    
    for perm in permutations(remaining):
        path = [start] + list(perm)
        d = _path_total_distance(path, dist_matrix)
        if d < best_dist:
            best_dist = d
            best_path = path
    
    return best_path


def _find_path_nearest_neighbor(start, cluster_ids, dist_matrix):
    """最近邻贪心，强制从start出发"""
    path = [start]
    visited = {start}
    
    while len(path) < len(cluster_ids):
        current = path[-1]
        nearest = min(
            (cid for cid in cluster_ids if cid not in visited),
            key=lambda cid: dist_matrix[(current, cid)]
        )
        path.append(nearest)
        visited.add(nearest)
    
    return path

