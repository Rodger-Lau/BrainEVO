
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
from tools.util import find_path
from tools.molecule_contrastive import compute_min_contrastive_distance

import matplotlib.pyplot as plt

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
        print(f"Saved {len(data_list)} molecules to {self.processed_paths[0]}")
def train_with_drawing(model, loader, optimizer, criterion, device, last_out, last_param, epoch, mode='acc'):
    grad_norms = {name: [] for name, param in model.named_parameters()}
    noise_rate = 0.01
    last_out = last_out.to(device)

    if mode == 'mape':
        last_param = 1-last_param
    last_param *= noise_rate
    model.train()

    total_loss = 0
    correct = 0
    total = 0
    activate_num = 0
    total_num = 0
    for data in loader:
        
        data.to(device)
        optimizer.zero_grad()
        out, activate_num = model(data, activate_num)
        total_num += 1
        if last_out.shape != data.y.shape:
            last_out = data.y
        target = last_param * last_out + (1-last_param) * data.y


        loss = criterion(out, target)
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms[name].append(grad_norm)
        optimizer.step()
        total_loss += loss.item() * data.num_graphs


        total += data.num_graphs
    return total_loss / total, activate_num / total_num,grad_norms
def draw_curve(grad_norms,save_dir,key,saved=True):

    plt.figure(figsize=(600,650))


    plt.plot(grad_norms, label='average gradient')
    plt.yscale('log')
    plt.title("Average gradient during training process")
    plt.legend()

    if saved:
        plt.savefig(os.path.join(save_dir, f'{key}_gradient.pdf'))
        with open(f"./saved_layer_gradient/HIV/train_sum_gradients_{key}.txt", 'w') as train_sum_gradient:
            train_sum_gradient.write(str(grad_norms.cpu()))

def train(model, loader, optimizer, criterion, device, last_out, last_param, epoch, mode='acc'):
    noise_rate = 0.01
    last_out = last_out.to(device)

    if mode == 'mape':
        last_param = 1-last_param
    last_param *= noise_rate
    model.train()

    total_loss = 0
    correct = 0
    total = 0
    activate_num = 0
    total_num = 0
    for data in loader:
        
        data.to(device)
        optimizer.zero_grad()
        out, activate_num = model(data, activate_num)
        total_num += 1
        if last_out.shape != data.y.shape:
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


if __name__ == "__main__":
    
    p0 = 0.5
    lambda0 = 1e-2

    torch.manual_seed(42)
    np.random.seed(42)
    zeta = 1.1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pos_weight = torch.tensor([20]).to(device)
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
    LTP_bool = True
    LTD_bool = True
    LTP = False
    LTD = False

    saved = False
    gradient_change_saved = False
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
        
        
        batch_size = min(32, len(train_idx) // 2)
        train_loader = DataLoader([dataset[i] for i in train_idx], batch_size=batch_size, shuffle=True)
        val_loader = DataLoader([dataset[i] for i in val_idx], batch_size=batch_size)
        test_loader = DataLoader([dataset[i] for i in test_idx], batch_size=batch_size)
        

        sample_data = dataset[0]
        in_channels = sample_data.x.shape[1]
        model = GIN(
            in_channels=in_channels,
            hidden_dim=128,
            num_layers=4,
            dropout=0.5
        ).to(device)


        for data in test_loader:
            output_shape = data.y.shape
            break
        

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10, verbose=True
        )
        print("Starting training...")
        

        best_val_auc = 0
        domain_results = {}
        
        for epoch in tqdm(range(1, first_stage_num_epochs+1)):
            train_loss, _ = train(model, train_loader, optimizer, criterion, device, last_out=torch.zeros(output_shape), last_param=0.0, epoch=epoch)
            if epoch % 10 == 0 or epoch == 1:
                test_loss, test_metrics, _ = evaluate(model, test_loader, criterion, device)
            if epoch == 100:
                    domain_results = test_metrics
        test_loss, test_metrics, _ = evaluate(model, test_loader, criterion, device)
        sum_grad, cat_grad = compute_gradient(model=model, train_loader=train_loader, device=device,criterion=criterion)
        sum_gradients.append(sum_grad)
        print(sum_gradients[domain_id])
        cat_grad_dict[domain_id] = cat_grad


    n_clusters = 2
 
    cluster_labels, cluster_members = clustering_dict(n_clusters=n_clusters, data_dict=cat_grad_dict, algorithm='kmeans')


    cluster_grad = get_cluster_average(cluster_members=cluster_members, cat_grad_dict=cat_grad_dict)
    min_grad, min_i = get_min_gradient(sum_gradients=cluster_grad,new_i=-1)

    common_model = GIN(
            in_channels=in_channels,
            hidden_dim=128,
            num_layers=4,
            dropout=0.5
        ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)


    sorted_clusters_gradients_list = find_path(cluster_grad)

    common_train_loss = []
    common_val_acc = []
    second_stage_circle = 3
    second_stage_num_epochs = 200
    new_i = 0
    sorted_sum_gradients = []
    sorted_cat_gradients = []
    train_cat_gradients = []
    train_sum_gradients = []
    for _ in range(second_stage_circle):
        train_number = 1
        for cluster_index in sorted_clusters_gradients_list:

            cluster = cluster_members[cluster_index]
            temp_sorted_gradients_list = sort_by_original_gradient(sum_gradients=[sum_gradients[domain] for domain in cluster])
            sorted_gradients_list = [cluster[index] for index in temp_sorted_gradients_list]
            for key in sorted_gradients_list:

                grad_norms = []
                sorted_sum_gradients.append(sum_gradients[key])
                sorted_cat_gradients.append(cat_grad_dict[key])
                if key == new_i : continue
                train_number -= 1
                if train_number>=0:  
                    if gradient_change_saved:
                        torch.save(common_model,f'./saved_models/HIV/common_model_{key}_starts.pth')
                    local_dataset = dataset[domains[key]]
                    weight_decay = my_sigmoid(lambda0,sum_gradients[key], sum_gradients[sorted_gradients_list[-1]])
                    optimizer = torch.optim.Adam(common_model.parameters(), lr=0.01, weight_decay=weight_decay)

                    common_model.drop_out = my_sigmoid(p0,sum_gradients[key], sum_gradients[sorted_gradients_list[-1]])


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
                    test_loss, test_metrics, _ = evaluate(common_model, val_loader, criterion, device)
                    test_acc = test_metrics['accuracy']
                    val_loss, val_metrics,_ = evaluate(common_model, test_loader, criterion, device)
                    val_acc = val_metrics['accuracy']
                    print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, '
                                f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
                    print(key,"开始训练...")
                    learning_rate = 0.001
                    for epoch in tqdm(range(1, second_stage_num_epochs+1)):
                        
                        train_loss, activate_frequency = train(common_model, train_loader, optimizer, criterion, device, last_out=last_output,
                                                                        last_param=test_acc,epoch=epoch)
                        sum_grad, cat_grad = compute_gradient(model=common_model, train_loader=train_loader, device=device,criterion=criterion)
                        grad_norms.append(sum_grad.cpu())
                        train_cat_gradients.append(cat_grad)
                        train_sum_gradients.append(sum_grad)
                        common_train_loss.append(train_loss)
                        if LTP and LTP_bool and saved:
                                    torch.save(common_model,'./saved_models1/LTP_post_common_model_HIV.pth')
                                    LTP_bool = False
                        if LTD and LTD_bool and saved:
                                    torch.save(common_model,'./saved_models1/LTD_post_common_model_HIV.pth')
                                    LTD_bool = False

                        test_loss, test_metrics, last_output = evaluate(common_model, val_loader, criterion, device)
                        test_acc = test_metrics['accuracy']
                        test_auc = test_metrics['auc']
                        val_loss, val_metrics,_ = evaluate(common_model, test_loader, criterion, device)
                        val_acc = val_metrics['accuracy']
                        val_auc = val_metrics['auc']
                        common_val_acc.append(val_acc)
                        if test_acc > best_acc:
                            best_acc = test_acc
                        if test_auc > best_auc:
                            best_auc = test_auc
                        print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, '
                                f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, '
                                f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
                        print_evaluation_metrics(val_metrics)
                        if epoch % 20 == 0:
                            if activate_frequency > 0.2:

                                if LTP_bool and saved:
                                    torch.save(common_model,'./saved_models1/LTP_pre_common_model_HIV.pth')
                                    LTP = True
                                    LTP_bool = False
                                weight_decay *= zeta
                                optimizer = torch.optim.Adam(common_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                                if common_model.drop_out*zeta<1:
                                    common_model.drop_out *= zeta
                                
                            elif activate_frequency<0.1:
                                if LTD_bool and saved:
                                    torch.save(common_model,'./saved_models1/LTD_pre_common_model_HIV.pth')
                                    
                                    LTD = True
                                    LTD_bool = False
                                weight_decay /=zeta
                                
                                optimizer = torch.optim.Adam(common_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                                common_model.drop_out/=zeta
                    print(f'最终测试准确率: {test_acc:.4f}, 最佳测试准确率: {best_acc:.4f},最终测试AUC: {test_auc:.4f}, 最佳测试AUC: {best_auc:.4f}')
                    print_evaluation_metrics(test_metrics)
                    draw_curve(grad_norms=grad_norms,save_dir='./gradient_visualization_HIV',key=key,saved=True)
                    if gradient_change_saved:
                            torch.save(common_model,f'./saved_models/HIV/common_model_{key}_ends.pth')
    best_acc = 0
    best_auc = 0.5
    coupled_train_loss = []
    coupled_val_acc = []
    local_dataset = dataset[domains[new_i]]
    domain_indices = domains[new_i]
    train_idx, test_idx = train_test_split(domain_indices, test_size=0.2, random_state=42)
    val_idx, test_idx = train_test_split(test_idx, test_size=0.5, random_state=42)
    train_loader = DataLoader([dataset[i] for i in train_idx], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader([dataset[i] for i in val_idx], batch_size=batch_size)
    test_loader = DataLoader([dataset[i] for i in test_idx], batch_size=batch_size)
    contrastive_train_indices = {new_i: train_idx}
    for domain_id, indices in domains.items():
        if domain_id == new_i or len(indices) < 10:
            continue
        source_train_idx, _ = train_test_split(indices, test_size=0.2, random_state=42)
        contrastive_train_indices[domain_id] = source_train_idx
    min_distance, nearest_domain = compute_min_contrastive_distance(
        common_model, dataset, contrastive_train_indices, new_i,
        batch_size, device,
    )
    kappa = 5

    fourth_stage_num_epochs = 200
    if min_distance < kappa:
        coupled_model = common_model
    else:
        coupled_model = GIN(
            in_channels=in_channels,
            hidden_dim=128,
            num_layers=4,
            dropout=0.5,
        ).to(device)
    coupled_model.to(device)
    coupled_learning_rate = 0.01
    coupled_weight_decay = 0.0001
    coupled_optimizer = torch.optim.Adam(coupled_model.parameters(), lr=coupled_learning_rate, weight_decay=coupled_weight_decay)
    for epoch in range(1, fourth_stage_num_epochs+1):
        train_loss, _ = train(coupled_model,train_loader,coupled_optimizer,criterion,device,
                                                        last_param=1,last_out=torch.zeros(output_shape),epoch=epoch,mode='acc')
        coupled_train_loss.append(train_loss)


        val_loss, val_results, out = evaluate(coupled_model, val_loader, criterion, device)
        val_acc = val_results['accuracy']
        coupled_val_acc.append(val_acc)
        test_loss, test_results, out = evaluate(coupled_model, test_loader, criterion, device)
        test_acc = test_results['accuracy']
        test_auc = test_results['auc']
        print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, '
                            f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, '
                            f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
        print_evaluation_metrics(val_results)
        if test_acc > best_acc:
                best_acc = test_acc
        if test_auc > best_auc:
            best_auc = test_auc
        
    print(f'最终测试准确率: {test_acc:.4f}, 最佳测试准确率: {best_acc:.4f},最终测试AUC: {test_auc:.4f}, 最佳测试AUC: {best_auc:.4f}')
    print_evaluation_metrics(test_results)
