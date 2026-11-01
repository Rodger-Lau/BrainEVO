import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d, Module, Dropout
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, MolFromSmiles
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import os
import shutil
from models.GIN_HIV import GIN
from tools.cluster import *
from tools.gradient_compute_HIV import *
import random
import argparse


class HIVDataset(InMemoryDataset):
    def __init__(self, root, csv_file, transform=None, pre_transform=None):
        self.csv_file = csv_file
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        return [os.path.basename(self.csv_file)]
    
    @property
    def processed_file_names(self):
        return ['data.pt']
    
    def download(self):
        pass
    
    def _has_carbonyl(self, mol):
        pass
        pattern = Chem.MolFromSmarts('[CX3]=[OX1]')
        return mol.HasSubstructMatch(pattern)
    
    def _has_halogen(self, mol):
        pass
        pattern = Chem.MolFromSmarts('[F,Cl,Br,I]')
        return mol.HasSubstructMatch(pattern)
    
    def process(self):

        df = pd.read_csv(os.path.join(self.raw_dir, self.raw_file_names[0]))
        

        if 'activity' in df.columns and 'HIV_active' not in df.columns:
            df.rename(columns={'activity': 'HIV_active'}, inplace=True)
        

        if 'HIV_active' not in df.columns:
            raise ValueError("CSV file must contain 'HIV_active' column")
        
        data_list = []
        

        domain_counts = {
            0: 0,
            1: 0,
            2: 0,
            3: 0
        }
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Molecules"):
            smiles = row['smiles']
            label = row['HIV_active']
            

            try:
                label = int(label)
            except (ValueError, TypeError):

                continue
                
            mol = Chem.MolFromSmiles(smiles)
            
            if mol is None:
                continue
                

            mol = Chem.AddHs(mol)
            

            has_carbonyl = self._has_carbonyl(mol)
            has_halogen = self._has_halogen(mol)
            

            domain_id = (1 if has_carbonyl else 0) * 2 + (1 if has_halogen else 0)
            domain_counts[domain_id] += 1
            

            atom_features = []
            for atom in mol.GetAtoms():
                features = [
                    atom.GetAtomicNum(),
                    atom.GetDegree(),
                    atom.GetImplicitValence(),
                    int(atom.GetIsAromatic()),
                    atom.GetTotalNumHs()
                ]
                atom_features.append(features)
            

            edge_index = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()

                edge_index.append([i, j])
                edge_index.append([j, i])
            

            try:
                x = torch.tensor(atom_features, dtype=torch.float)
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                y = torch.tensor([label], dtype=torch.float)
                

                data = Data(
                    x=x, 
                    edge_index=edge_index, 
                    y=y,
                    domain=torch.tensor([domain_id], dtype=torch.long)
                )
                data_list.append(data)
            except Exception as e:
                print(f"Error processing SMILES: {smiles}, error: {e}")
                continue
        



        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
            
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
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
        out, activate_num = model(data)
        total_num += 1
        if last_out.shape!=data.y.shape:
            last_out = data.y

        target = data.y
        loss = criterion(out, target)
        loss.backward()

        optimizer.step()
        total_loss += loss.item() * data.num_graphs


        total += data.num_graphs
    return total_loss / total, activate_num / total_num


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold=0.5):
    model.eval()
    total_loss = 0
    y_true, y_prob = [], []
    
    for data in loader:
        data = data.to(device)
        out,_ = model(data)
        target = data.y

        loss = criterion(out, target)


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
        results['auc'] = 0.5
    
    try:
        results['accuracy'] = accuracy_score(y_true, y_pred)
    except:
        print('error')
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
    



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelname', type=str, default = None)
    args = parser.parse_args()
    model_name = args.modelname
    p0 = 0.5
    lambda0 = 1e-2

    torch.manual_seed(42)
    np.random.seed(42)
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pos_weight = torch.tensor([60]).to(device)
    print(f"Using device: {device}")
    

    data_dir = './data/hiv'
    os.makedirs(data_dir, exist_ok=True)
    

    src_csv = './data/HIV.csv'
    dst_csv = os.path.join(data_dir, 'raw', os.path.basename(src_csv))
    os.makedirs(os.path.dirname(dst_csv), exist_ok=True)
    if not os.path.exists(dst_csv):
        shutil.copy(src_csv, dst_csv)
        print(f"Copied CSV file to {dst_csv}")
    

    print("Loading HIV dataset...")
    dataset = HIVDataset(root=data_dir, csv_file=src_csv)
    num_attributes = 2

    domains = {
        0:None, 1:None, 2:None, 3:None
    }
    

    all_results = {}
    sum_gradients = []
    cat_grad_dict = {}
    first_stage_num_epochs = 200
    random_task_order = np.arange(len(domains))
    random.shuffle(random_task_order)
    for domain_id in domains:

        print(f"Training model for Domain {domain_id}:")
        domain_indices = []
        for i in range(len(dataset)):
            data = dataset[i]
            if torch.equal(data.domain, torch.tensor([domain_id], dtype=torch.long)):
                domain_indices.append(i)
        domains[domain_id] = domain_indices        
        train_idx, test_val_idx = train_test_split(domain_indices, test_size=0.2, random_state=42)
        val_idx, test_idx = train_test_split(test_val_idx, test_size=0.5, random_state=42)
        batch_size = 64


        test_loader = DataLoader([dataset[i] for i in test_idx], batch_size=batch_size)
        

        sample_data = dataset[0]
        in_channels = sample_data.x.shape[1]
        for data in test_loader:
            output_shape = data.y.shape
            break
    if model_name == 'GIN':
            
        common_model = GIN(
                in_channels=in_channels,
                hidden_dim=128,
                num_layers=4,
                dropout=0.5
            ).to(device)
        

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)


    optimizer = torch.optim.Adam(common_model.parameters(), lr=0.01, weight_decay=1e-4)

    common_train_loss = []
    common_val_acc = []
    second_stage_circle = 5
    second_stage_num_epochs = 200
    new_i = 0
    total_auc = 0
    total_best_auc = 0
    total_acc = 0
    total_best_acc = 0
    for __ in range(3):
        random.shuffle(random_task_order)
        for _ in range(second_stage_circle):
            train_number = 1
            for key in random_task_order:
                    if key == new_i : continue
                    train_number-=1
                    if train_number>=0:
                        domain_indices = domains[key]
                        train_idx, test_idx = train_test_split(domain_indices, test_size=0.2, random_state=42)
                        val_idx, test_idx = train_test_split(test_idx, test_size=0.5, random_state=42)
                    

                        batch_size = 64
                        train_loader = DataLoader([dataset[i] for i in train_idx], batch_size=batch_size, shuffle=True)
                        val_loader = DataLoader([dataset[i] for i in val_idx], batch_size=batch_size)
                        test_loader = DataLoader([dataset[i] for i in test_idx], batch_size=batch_size)
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                        best_auc = 0.5
                        test_acc = 0.0
                        best_acc = 0.0
                        last_output = torch.zeros(output_shape)
                        print(key,"开始训练...")
                        learning_rate = 0.001
                        for epoch in tqdm(range(1, second_stage_num_epochs+1)):
                            train_loss, activate_frequency = train(common_model, train_loader, optimizer, criterion, device, last_out=last_output,
                                                                            last_param=test_acc,epoch=epoch)

                            test_loss, test_metrics, last_output = evaluate(common_model, val_loader, criterion, device)
                            test_acc = test_metrics['accuracy']
                            test_auc = test_metrics['auc']
                            val_loss, val_metrics,_ = evaluate(common_model, test_loader, criterion, device)
                            val_acc = val_metrics['accuracy']
                            val_auc = val_metrics['auc']
                            if test_acc > best_acc:
                                best_acc = test_acc
                            if test_auc > best_auc:
                                best_auc = test_auc
                            if epoch % 20 == 0:
                                print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, '
                                    f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, '
                                    f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
                                print_evaluation_metrics(val_metrics)
                        print(f'最终测试准确率: {test_acc:.4f}, 最佳测试准确率: {best_acc:.4f},最终测试AUC: {test_auc:.4f}, 最佳测试AUC: {best_auc:.4f}')
                        print_evaluation_metrics(test_metrics)

        best_acc = 0
        coupled_train_loss = []
        coupled_val_acc = []
        local_dataset = dataset[domains[new_i]]
        domain_indices = domains[new_i]
        train_idx, test_idx = train_test_split(domain_indices, test_size=0.2, random_state=42)
        val_idx, test_idx = train_test_split(test_idx, test_size=0.5, random_state=42)
        train_loader = DataLoader([dataset[i] for i in train_idx], batch_size=batch_size, shuffle=True)
        val_loader = DataLoader([dataset[i] for i in val_idx], batch_size=batch_size)
        test_loader = DataLoader([dataset[i] for i in test_idx], batch_size=batch_size)

        fourth_stage_num_epochs = 200
        coupled_model = common_model
        coupled_model.to(device)
        coupled_learning_rate = 0.01
        coupled_weight_decay = 0.0001
        coupled_optimizer = torch.optim.Adam(coupled_model.parameters(), lr=coupled_learning_rate, weight_decay=coupled_weight_decay)
        best_acc = 0
        best_auc = 0.5
        test_acc = 0
        for epoch in range(1, fourth_stage_num_epochs+1):
            los, _ = train(coupled_model,train_loader,coupled_optimizer,criterion,device,
                                                            last_param=0,last_out=torch.zeros(output_shape),epoch=epoch,mode='acc')
            coupled_train_loss.append(los)


            val_loss, val_results, out = evaluate(coupled_model, val_loader, criterion, device)
            val_acc = val_results['accuracy']
            val_auc = val_results['auc']
            coupled_val_acc.append(val_acc)
            test_loss, test_results, out = evaluate(coupled_model, test_loader, criterion, device)
            test_acc = test_results['accuracy']
            test_auc = test_results['auc']
            print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, '
                                f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, '
                                f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
            if test_acc > best_acc:
                best_acc = test_acc
            if best_auc > test_auc:
                best_auc = test_auc
        print(f'最终测试准确率: {test_acc:.4f}, 最佳测试准确率: {best_acc:.4f},最终测试AUC: {test_auc:.4f}, 最佳测试AUC: {best_auc:.4f}')
        total_auc += test_auc
        total_best_auc += best_auc
        total_acc += test_acc
        total_best_acc += best_acc
    print(f'最终平均测试准确率: {total_acc/3:.4f}, 平均最佳测试准确率: {total_best_acc/3:.4f},最终平均测试AUC: {total_auc/3:.4f}, 平均最佳测试AUC: {total_best_auc/3:.4f}')
