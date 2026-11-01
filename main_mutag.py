import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_add_pool
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
from rdkit import Chem

from tqdm import tqdm
from tools.gradient_compute_mutag import *
from models.GIN_mutag import GIN
from tools.cluster import *
from tools.util import find_path
import random
from molecule_test_sampling import (
    BOOTSTRAP_SEEDS, aggregate_group_metrics, split_domain_indices,
)
from tools.molecule_contrastive import compute_min_contrastive_distance

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
class MutagDatasetWithSubstructures(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        return ['MUTAG_A.txt', 'MUTAG_graph_indicator.txt', 
                'MUTAG_graph_labels.txt', 'MUTAG_node_labels.txt']
    
    @property
    def processed_file_names(self):
        return ['mutag_data_with_substructures.pt']
    
    def download(self):
        pass
    
    def process(self):


        graph_indicator = []
        with open(os.path.join(self.raw_dir, 'MUTAG_graph_indicator.txt'), 'r') as f:
            for line in f:
                graph_indicator.append(int(line.strip()))
        

        node_labels = []
        with open(os.path.join(self.raw_dir, 'MUTAG_node_labels.txt'), 'r') as f:
            for line in f:
                node_labels.append(int(line.strip()))
        

        graph_labels = []
        with open(os.path.join(self.raw_dir, 'MUTAG_graph_labels.txt'), 'r') as f:
            for line in f:

                label = int(line.strip())
                graph_labels.append(0 if label == 1 else 1)
        

        edges = []
        with open(os.path.join(self.raw_dir, 'MUTAG_A.txt'), 'r') as f:
            for line in f:
                src, dst = map(int, line.strip().split(','))
                edges.append((src, dst))
        

        num_graphs = max(graph_indicator)
        data_list = []
        

        graph_nodes = {}
        for node_id, graph_id in enumerate(graph_indicator, start=1):
            if graph_id not in graph_nodes:
                graph_nodes[graph_id] = []
            graph_nodes[graph_id].append(node_id)
        

        for graph_id in tqdm(range(1, num_graphs + 1), desc="Processing graphs"):

            node_ids = graph_nodes[graph_id]
            num_nodes = len(node_ids)
            

            node_features = torch.zeros(num_nodes, 7)
            atom_types = []
            for i, node_id in enumerate(node_ids):
                label = node_labels[node_id - 1]
                node_features[i, label] = 1
                atom_types.append(label)
            

            edge_index = []
            for src, dst in edges:
                if src in node_ids and dst in node_ids:

                    local_src = node_ids.index(src)
                    local_dst = node_ids.index(dst)
                    edge_index.append([local_src, local_dst])
                    edge_index.append([local_dst, local_src])
            
            if len(edge_index) == 0:

                for i in range(num_nodes):
                    edge_index.append([i, i])
            
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            

            y = torch.tensor([graph_labels[graph_id - 1]], dtype=torch.float)
            

            data = Data(x=node_features, edge_index=edge_index, y=y)
            has_carbonyl = False
            has_halogen = False
            if any(atom_type in [3, 4, 5, 6] for atom_type in atom_types):
                has_halogen = True
            if 0 in atom_types and 2 in atom_types:

                carbon_indices = [i for i, at in enumerate(atom_types) if at == 0]
                oxygen_indices = [i for i, at in enumerate(atom_types) if at == 2]
                

                edge_list = edge_index.t().tolist()
                for c_idx in carbon_indices:
                    for o_idx in oxygen_indices:
                        if [c_idx, o_idx] in edge_list or [o_idx, c_idx] in edge_list:
                            has_carbonyl = True
                            break
                    if has_carbonyl:
                        break


            data.has_carbonyl = torch.tensor([int(has_carbonyl)])
            data.has_halogen = torch.tensor([int(has_halogen)])

            data_list.append(data)
        

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
    
    def reconstruct_molecule(self, atom_types, edge_index):
        pass

        atom_map = {
            0: 'C',
            1: 'N',
            2: 'O',
            3: 'F',
            4: 'I',
            5: 'Cl',
            6: 'Br'
        }
        

        mol = Chem.RWMol()
        

        atom_idx_map = {}
        for i, atom_type in enumerate(atom_types):
            symbol = atom_map.get(atom_type, 'C')
            atom = Chem.Atom(symbol)
            idx = mol.AddAtom(atom)
            atom_idx_map[i] = idx
        

        edge_list = edge_index.t().tolist()
        added_bonds = set()
        
        for src, dst in edge_list:

            if src == dst:
                continue
            

            bond_key = tuple(sorted((src, dst)))
            if bond_key in added_bonds:
                continue
            added_bonds.add(bond_key)
            

            mol.AddBond(atom_idx_map[src], atom_idx_map[dst], Chem.BondType.SINGLE)
        

        try:
            mol = mol.GetMol()

            Chem.SanitizeMol(mol)
            return mol
        except Exception:
            return None
def train(model, loader, optimizer, criterion, device, last_out, last_param, epoch, mode='acc'):
    
    last_out = last_out.to(device)

    if mode == 'mape':
        last_param = 1-last_param
    last_param *= 0.01
    model.train()

    total_loss = 0
    correct = 0
    total = 0
    activate_num = 0
    total_num = 0
    for data in loader:
        
        data.to(device)
        optimizer.zero_grad()
        out, activate_num = model(data.x, data.edge_index, data.batch, activate_num)
        total_num += 1
        if last_out.shape!=data.y.shape:
            last_out = data.y
        target = last_param * last_out + (1-last_param) * data.y
        target = data.y


        if out.shape!=target.shape:
            out = out.unsqueeze(dim=0)

        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        pred = (out >= 0.5).cpu().numpy().astype(float)

        correct += (pred == data.y.cpu().numpy()).sum().item()
        total += data.num_graphs
    
    return total_loss / total, correct / total, activate_num / total_num

@torch.no_grad()
def validate(model, loader, criterion,device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for data in loader:
        data.to(device)
        out,_ = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        
        total_loss += loss.item() * data.num_graphs

        pred = (out >= 0.5).cpu().numpy().astype(float)
        correct += (pred == data.y.cpu().numpy()).sum().item()
        total += data.num_graphs
    
    return total_loss / total, correct / total
@torch.no_grad()
def test(model, loader, criterion,device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for data in loader:
        data.to(device)
        out,_ = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        
        total_loss += loss.item() * data.num_graphs
        

        pred = (out >= 0.5).cpu().numpy().astype(float)
        correct += (pred == data.y.cpu().numpy()).sum().item()
        total += data.num_graphs
    
    return total_loss / total, correct / total, out

@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold=0.5):
    model.eval()
    total_loss = 0
    y_true, y_prob = [], []
    
    for data in loader:
        data = data.to(device)
        out,_ = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        total_loss += loss.item() * data.num_graphs
        

        y_true.append(data.y.cpu())
        y_prob.append(out.sigmoid().detach().cpu())
    
    y_true = torch.cat(y_true).numpy()
    y_prob = torch.cat(y_prob).numpy()
    

    y_pred = (y_prob >= threshold).astype(int)
    results = {}
    
    try:
        results['auc'] = roc_auc_score(y_true, y_prob)
    except ValueError:
        print('auc error')
        results['auc'] = 0.5
    
    try:
        results['accuracy'] = accuracy_score(y_true, y_pred)
    except:
        results['accuracy'] = 0.0
        
    try:
        results['precision'] = precision_score(y_true, y_pred, zero_division=0)
    except:
        results['precision'] = 0.0
        
    try:
        results['recall'] = recall_score(y_true, y_pred, zero_division=0)
    except:
        results['recall'] = 0.0
        
    try:
        results['f1'] = f1_score(y_true, y_pred)
    except:
        results['f1'] = 0.0
    

    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        results['confusion_matrix'] = {
            'true_negative': tn,
            'false_positive': fp,
            'false_negative': fn,
            'true_positive': tp
        }
    except:
        results['confusion_matrix'] = None
    
    return total_loss / len(loader.dataset), results, out

def print_evaluation_metrics(metrics, prefix=""):
    print(f"{prefix}AUC: {metrics['auc']:.4f}")
    print(f"{prefix}Accuracy: {metrics['accuracy']:.4f}")
    print(f"{prefix}Precision: {metrics['precision']:.4f}")
    print(f"{prefix}Recall: {metrics['recall']:.4f}")
    print(f"{prefix}F1 Score: {metrics['f1']:.4f}")


def balanced_training_ids(train_ids, target_size, seed):
    pass
    ids = list(train_ids)
    if ids and len(ids) < target_size:
        ids.extend(random.Random(seed).choices(ids, k=target_size - len(ids)))
    return ids


@torch.no_grad()
def evaluate_bootstrap_groups(model, loaders, criterion, device):
    group_rows = []
    output = None
    for seed, loader in zip(BOOTSTRAP_SEEDS, loaders):
        loss, metrics, output = evaluate(model, loader, criterion, device)
        group_rows.append({"seed": seed, "loss": loss, **metrics})
    summary = aggregate_group_metrics(group_rows)
    mean_metrics = {
        key: values["mean"]
        for key, values in summary.items()
    }
    mean_metrics["confusion_matrix"] = None
    mean_metrics["bootstrap_groups"] = group_rows
    mean_loss = summary.get("loss", {"mean": 0.0})["mean"]
    return mean_loss, mean_metrics, output

def parse_bool(value):
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("expected True or False")

def make_test_loaders(dataset, split, batch_size, bool_bootstrap):
    if bool_bootstrap:
        test_indices = [split["test_groups"][seed] for seed in BOOTSTRAP_SEEDS]
    else:
        test_indices = [split["test"]]
    return [
        DataLoader(dataset[indices], batch_size=batch_size, shuffle=False)
        for indices in test_indices
    ]

def evaluate_test_sets(model, loaders, criterion, device, bool_bootstrap):
    if bool_bootstrap:
        return evaluate_bootstrap_groups(model, loaders, criterion, device)
    return evaluate(model, loaders[0], criterion, device)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bool_bootstrap", type=parse_bool, default=False)
    bool_bootstrap = parser.parse_args().bool_bootstrap
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    setup_seed(42)
    zeta = 1.1
    p0 = 0.5
    lambda0 = 1e-4
    batch_size = 8

    dataset = MutagDatasetWithSubstructures(root='./data/MUTAG1/MUTAG')
    domains = {
        0: [],
        1: [],
        2: [],
        3: []
    }
    domains={}
    for i in range(4):
        domains[i]=[]
    sum_gradients = []
    

    for idx in range(len(dataset)):
        data = dataset[idx]


        has_carbonyl = data.has_carbonyl.item()
        has_halogen = data.has_halogen.item()
        
        if has_carbonyl and has_halogen:
            domains[0].append(idx)
        elif has_carbonyl and not has_halogen:
            domains[1].append(idx)
        elif not has_carbonyl and has_halogen:
            domains[2].append(idx)
        else:
            domains[3].append(idx)
    domain_splits = split_domain_indices(
        domains, split_seed=42, bool_bootstrap=bool_bootstrap
    )
    max_train_length = max(
        len(domain_splits[key]["train"]) for key in domains
    )
    cat_grad_dict = {}
    

    first_stage_num_epochs = 200
    for key in domains:
        split = domain_splits[key]
        train_ids = balanced_training_ids(
            split["train"], max_train_length, seed=42 + key
        )
        train_dataset = dataset[train_ids]
        val_dataset = dataset[split["val"]]
        test_loaders = make_test_loaders(
            dataset, split, batch_size, bool_bootstrap
        )
        test_loader = test_loaders[0]
        

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        for data in test_loader:
            output_shape = data.y.shape
            break
        drop_out = 0.5
        model = GIN(hidden_dim=64, num_layers=3, drop_out=0.5).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
        print(key,"开始训练...")
        for epoch in range(1, first_stage_num_epochs+1):
            train_loss, train_acc, _ = train(model, train_loader, optimizer, criterion, device, last_out=torch.zeros(output_shape), last_param=0.0, epoch=epoch)
        sum_grad, cat_grad = compute_gradient(model=model, train_loader=train_loader, device=device,criterion=criterion)
        sum_gradients.append(sum_grad)
        cat_grad_dict[key] = cat_grad

    n_clusters = 2
 
    cluster_labels, cluster_members = clustering_dict(n_clusters=n_clusters, data_dict=cat_grad_dict, algorithm='kmeans')
    cluster_grad = get_cluster_average(cluster_members=cluster_members, cat_grad_dict=cat_grad_dict)


    min_grad, min_i = get_min_gradient(sum_gradients=cluster_grad,new_i=-1)

    common_model = GIN(hidden_dim=64, num_layers=3, drop_out=0.5).to(device)
    criterion = nn.BCEWithLogitsLoss()
    sorted_clusters_gradients_list = find_path(cluster_grad)


    common_train_loss = []
    common_val_acc = []
    second_stage_circle = 5
    second_stage_num_epochs = 200
    new_i = 0
    for _ in range(second_stage_circle):
        for cluster_index in sorted_clusters_gradients_list:
            train_number=1

            cluster = cluster_members[cluster_index]
            temp_sorted_gradients_list = sort_by_original_gradient(sum_gradients=[sum_gradients[domain] for domain in cluster])
            sorted_gradients_list = [cluster[index] for index in temp_sorted_gradients_list]
            for key in sorted_gradients_list:
                if key == new_i : continue
                train_number-=1
                if train_number>=0:
                    weight_decay = my_sigmoid(lambda0,sum_gradients[key], sum_gradients[sorted_gradients_list[-1]])
                    optimizer = torch.optim.Adam(common_model.parameters(), lr=0.01, weight_decay=weight_decay)

                    common_model.drop_out = my_sigmoid(p0,sum_gradients[key], sum_gradients[sorted_gradients_list[-1]])

                    split = domain_splits[key]
                    train_ids = balanced_training_ids(
                        split["train"], max_train_length, seed=42 + key
                    )
                    train_dataset = dataset[train_ids]
                    val_dataset = dataset[split["val"]]
                    test_loaders = make_test_loaders(
                        dataset, split, batch_size, bool_bootstrap
                    )
                    test_loader = test_loaders[0]
                    

                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                    test_acc = 0.0
                    best_acc = 0.0
                    best_auc = 0.5
                    last_output = torch.zeros(output_shape)
                    print(key,"开始训练...")
                    
                    for epoch in range(1, second_stage_num_epochs+1):
                        train_loss, train_acc, activate_frequency = train(common_model, train_loader, optimizer, criterion, device, last_out=last_output,
                                                                        last_param=test_acc,epoch=epoch)
                        common_train_loss.append(train_loss)

                        test_loss, test_acc, last_output = test(common_model, val_loader, criterion, device)


                        val_loss, val_metrics, val_output = evaluate(common_model, val_loader, criterion, device)
                        test_loss, test_metrics, test_output = evaluate_test_sets(
                            common_model, test_loaders, criterion, device,
                            bool_bootstrap,
                        )
                        val_acc = val_metrics['accuracy']
                        test_acc = test_metrics['accuracy']
                        test_auc = test_metrics['auc']
                        common_val_acc.append(val_acc)
                        if test_acc > best_acc:
                            best_acc = test_acc
                        if test_auc > best_auc:
                            best_auc = test_auc
                        
                        if epoch % 20 == 0:
                            if activate_frequency > 0.8:
                                weight_decay *= zeta
                                optimizer = torch.optim.Adam(common_model.parameters(), lr=0.01, weight_decay=weight_decay)
                                common_model.drop_out *= zeta
                            elif activate_frequency<0.2:
                                weight_decay /= zeta
                                optimizer = torch.optim.Adam(common_model.parameters(), lr=0.01, weight_decay=weight_decay)
                                common_model.drop_out/=zeta
                            print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                                f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, '
                                f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
                            print_evaluation_metrics(val_metrics)
                    print(f'最终测试准确率: {test_acc:.4f}, 最佳测试准确率: {best_acc:.4f},最终测试AUC: {test_auc:.4f}, 最佳测试AUC: {best_auc:.4f}')
                    print_evaluation_metrics(test_metrics)
    best_acc = 0
    coupled_train_loss = []
    coupled_val_acc = []
    split = domain_splits[new_i]
    train_ids = balanced_training_ids(
        split["train"], max_train_length, seed=42 + new_i
    )
    train_dataset = dataset[train_ids]
    val_dataset = dataset[split["val"]]
    test_loaders = make_test_loaders(
        dataset, split, batch_size, bool_bootstrap
    )
    test_loader = test_loaders[0]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    contrastive_train_indices = {
        domain_id: balanced_training_ids(
            domain_splits[domain_id]["train"], max_train_length,
            seed=42 + domain_id,
        )
        for domain_id in domain_splits
    }
    min_distance, nearest_domain = compute_min_contrastive_distance(
        common_model, dataset, contrastive_train_indices, new_i,
        batch_size, device,
    )
    kappa = 5


    fourth_stage_num_epochs = 200
    if min_distance < kappa:
        coupled_model = common_model
    else:
        coupled_model = GIN(hidden_dim=64, num_layers=3, drop_out=0.5).to(device)
    coupled_model.to(device)
    coupled_learning_rate = 0.01
    coupled_weight_decay = 0.0001
    coupled_optimizer = torch.optim.Adam(coupled_model.parameters(), lr=coupled_learning_rate, weight_decay=coupled_weight_decay)
    for epoch in range(1, fourth_stage_num_epochs+1):
        train_loss, train_acc, _ = train(coupled_model,train_loader,coupled_optimizer,criterion,device,
                                                        last_param=1,last_out=torch.zeros(output_shape),epoch=epoch,mode='acc')
        coupled_train_loss.append(train_loss)


        val_loss, val_acc = validate(coupled_model, val_loader, criterion, device)
        coupled_val_acc.append(val_acc)
        val_loss, val_metrics, val_output = evaluate(coupled_model, val_loader, criterion, device)
        test_loss, test_metrics, test_output = evaluate_test_sets(
            coupled_model, test_loaders, criterion, device,
            bool_bootstrap,
        )
        test_acc = test_metrics["accuracy"]
        test_auc = test_metrics["auc"]
        if test_acc > best_acc:
            best_acc = test_acc
        if test_auc > best_auc:
            best_auc = test_auc
        print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                            f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}'
                            f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
                            
        print_evaluation_metrics(val_metrics)
    print(f'最终测试准确率: {test_acc:.4f}, 最佳测试准确率: {best_acc:.4f},最终测试AUC: {test_auc:.4f}, 最佳测试AUC: {best_auc:.4f}')
    print_evaluation_metrics(test_metrics)
if __name__ == "__main__":
    main()
