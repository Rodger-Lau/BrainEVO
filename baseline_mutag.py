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
from rdkit.Chem import Draw, AllChem
from tqdm import tqdm
from tools.gradient_compute_mutag import *
from models.GIN_mutag import GIN
from tools.cluster import *
import random
import argparse


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
            

            y = torch.tensor([graph_labels[graph_id - 1]], dtype=torch.long)
            

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
        if out.shape!=target.shape:
            out = out.unsqueeze(dim=0)
        target = data.y
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelname', type=str, default = None)
    args = parser.parse_args()
    model_name = args.modelname
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    setup_seed(42)
    p0 = 0.5
    lambda0 = 1e-2
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
    max_length = max(len(domains[key]) for key in domains)
    cat_grad_dict = {}

    first_stage_num_epochs = 200
    for key in domains:
        length = len(domains[key])
        for i in range(max_length//length-1):
            for j in range(length):
                domains[key].append(domains[key][j])
        local_dataset = dataset[domains[key]]
        test_indices = list(range(int(0.9*len(local_dataset)), len(local_dataset)))
        test_dataset = local_dataset[test_indices]
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        for data in test_loader:
            output_shape = data.y.shape
            break
    in_channels = 7
    if model_name == 'GIN':
        common_model = GIN(hidden_dim=64, num_layers=3, drop_out=0.5).to(device)

    criterion = nn.BCEWithLogitsLoss()
    
    optimizer = torch.optim.Adam(common_model.parameters(), lr=0.01, weight_decay=1e-4)
 
    common_train_loss = []
    common_val_acc = []
    second_stage_circle = 5
    second_stage_num_epochs = 200
    new_i = 0
    total_acc = 0
    total_best_acc = 0
    total_auc = 0
    total_best_auc = 0
    for __ in range(3):
        for _ in range(second_stage_circle):
                train_number = 2
                for key in domains:
                    if key == new_i : continue
                    train_number -= 1
                    if train_number>=0:

                        local_dataset = dataset[domains[key]]
                        train_indices = list(range(int(0.7*len(local_dataset))))
                        val_indices = list(range(int(0.7*len(local_dataset)),int(0.9*len(local_dataset))))
                        test_indices = list(range(int(0.9*len(local_dataset)), len(local_dataset)))
                        train_dataset = local_dataset[train_indices]
                        val_dataset = local_dataset[val_indices]
                        test_dataset = local_dataset[test_indices]
                        

                        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                        best_auc = 0.5
                        test_acc = 0.0
                        best_acc = 0.0
                        last_output = torch.zeros(output_shape)
                        print(key,"开始训练...")
                        
                        for epoch in range(1, second_stage_num_epochs+1):
                            train_loss, train_acc, activate_frequency = train(common_model, train_loader, optimizer, criterion, device, last_out=last_output,
                                                                            last_param=test_acc,epoch=epoch)
                            common_train_loss.append(train_loss)


                            test
                            val_loss, val_metrics, val_output = evaluate(common_model, val_loader, criterion, device)
                            test_loss, test_metrics, test_output = evaluate(common_model, test_loader,criterion, device)
                            val_acc = val_metrics['accuracy']
                            test_acc = test_metrics['accuracy']
                            test_auc = test_metrics['auc']

                            common_val_acc.append(val_acc)
                            if test_acc > best_acc:
                                best_acc = test_acc
                            if test_auc > best_auc:
                                best_auc = test_auc
                            
                            if epoch % 20 == 0:
                                print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                                    f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, '
                                    f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
                                print_evaluation_metrics(val_metrics)
                        print(f'最终测试准确率: {test_acc:.4f}, 最佳测试准确率: {best_acc:.4f},最终测试AUC: {test_auc:.4f}, 最佳测试AUC: {best_auc:.4f}')
                        print_evaluation_metrics(test_metrics)
        best_acc = 0
        coupled_train_loss = []
        coupled_val_acc = []
        local_dataset = dataset[domains[new_i]]
        train_indices = list(range(int(0.7*len(local_dataset))))
        val_indices = list(range(int(0.7*len(local_dataset)),int(0.9*len(local_dataset))))
        test_indices = list(range(int(0.9*len(local_dataset)), len(local_dataset)))
        train_dataset = local_dataset[train_indices]
        val_dataset = local_dataset[val_indices]
        test_dataset = local_dataset[test_indices]
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        fourth_stage_num_epochs = 200
        coupled_model = common_model
        coupled_model.to(device)
        coupled_learning_rate = 0.01
        coupled_weight_decay = 0.0001
        coupled_optimizer = torch.optim.Adam(coupled_model.parameters(), lr=coupled_learning_rate, weight_decay=coupled_weight_decay)
        best_auc = 0.5
        test_acc = 0.0
        best_acc = 0.0
        for epoch in range(1, fourth_stage_num_epochs+1):
            los, ac, act_num = train(coupled_model,train_loader,coupled_optimizer,criterion,device,
                                                            last_param=0,last_out=torch.zeros(output_shape),epoch=epoch,mode='acc')
            coupled_train_loss.append(los)


            val_loss, val_acc = validate(coupled_model, val_loader, criterion, device)
            coupled_val_acc.append(val_acc)
            test_loss, test_acc, test_out=test(coupled_model, test_loader, criterion, device)
            val_loss, val_metrics, val_output = evaluate(common_model, val_loader, criterion, device)
            test_loss, test_metrics, test_output = evaluate(common_model, test_loader,criterion, device)
            test_auc = test_metrics['auc']
            
            print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                                f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}'
                                f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
            print_evaluation_metrics(val_metrics)
            if test_acc > best_acc:
                best_acc = test_acc
            if test_auc > best_auc:
                best_auc = test_auc
        print(f'最终测试准确率: {test_acc:.4f}, 最佳测试准确率: {best_acc:.4f},最终测试AUC: {test_auc:.4f}, 最佳测试AUC: {best_auc:.4f}')
        print_evaluation_metrics(test_metrics)
        total_acc += test_acc
        total_best_acc += best_acc
        total_auc += test_auc
        total_best_auc += best_auc
    print(f'最终平均测试准确率: {total_acc/3:.4f}, 平均最佳测试准确率: {total_best_acc/3:.4f},最终平均测试AUC：{total_auc/3:.4f}, 平均最佳测试AUC: {total_best_auc/3:.4f}')
if __name__ == "__main__":
    main()
