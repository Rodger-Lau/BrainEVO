import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


from tools.metrics import *
from models.gwn import gwnet
from models.continual_baselines import build_baseline
from tools.dataloader import *
from tools.gradient_compute import *
from tools.util import choose_sublist
import os
from torch.utils.data import Dataset
import torch.optim as optim
from engine import *
from tools.cluster_ST import *
import argparse
import matplotlib.pyplot as plt
import logging
import builtins
from datetime import datetime
def draw_curve(grad_norms,save_dir,key,circle,saved=True):

    plt.figure(figsize=(600,650))
    for name, norms in grad_norms.items():
        plt.plot(norms, label=name)
    plt.yscale('log')
    plt.title("Gradient Norms per Layer")
    plt.legend()

    if saved:
        plt.savefig(os.path.join(save_dir, f'circle{circle}_{key[0]}_{key[1]}_gradient.pdf'))
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default = None)
parser.add_argument('--baseline', choices=['original','pnn','hidepet'], default='original')
args = parser.parse_args()
log_dir = os.path.join('logs', args.baseline)
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'{args.dataset}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
logger = logging.getLogger('baseline_ST_new')
logger.setLevel(logging.INFO)
logger.handlers.clear()
handler = logging.FileHandler(log_file, encoding='utf-8')
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
def _log_print(*values, **kwargs):
    logger.info(' '.join(str(v) for v in values))
builtins.print = _log_print
logger.info('baseline=%s dataset=%s', args.baseline, args.dataset)
dataset_path = args.dataset

if dataset_path == 'CHI':
    tasks = ['RISK','TAXIDROP','TAXIPICK']
elif dataset_path == 'NYC':
    tasks = ['CROWDIN','CROWDOUT','TAXIDROP','TAXIPICK']
else:
    tasks = ['FLOW','SPEED']

dataset_path = os.path.join('./data/data',dataset_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

task_dirs = [os.path.join(dataset_path, item) for item in tasks]
train_loaders = {}
val_loaders = {}
test_loaders = {}
task_per_dir = 4
for i in range(len(task_dirs)):
    train_loaders[i] = {}
    val_loaders[i] = {}
    test_loaders[i] = {}
in_dim = 3
scalers=[]
batch_size = 32

first_stage_num_epochs = 30
for i in range(len(task_dirs)):
    dataloaders, scaler, num_nodes = get_dataloaders_scaler_and_split_task(task_dirs[i], batch_size,task_per_dir=task_per_dir)

    for j in range(len(dataloaders)):
        train_loaders[i][j]=(dataloaders[j]['train'])
        val_loaders[i][j]=(dataloaders[j]['val'])
        test_loaders[i][j]=(dataloaders[j]['test'])
    scalers.append(scaler)
new_i = 1
if args.dataset == 'NYC': new_i=3
new_j = len(train_loaders[0])-1
new_train_loader = train_loaders[new_i][new_j]
new_val_loader = val_loaders[new_i][new_j]
new_test_loader = test_loaders[new_i][new_j]
for data in new_test_loader:
    inputs, reals = data
    output_shape = reals.shape
new_scaler = scalers[new_i][new_j]
# Controlling cross-source experiments, no_task = True when conducting cross-source experiments
no_task = True
# Controlling cross-temporal experiments, no_temporal = True when conducting cross-temporal experiments
no_temporal = False
assert no_task != no_temporal
sum_gradients = []
cat_gradients = []
print('we skip task %d for %s'%(new_j,task_dirs[new_i]))

for i in range(len(train_loaders)):
    if no_task and i == new_i:
        continue
    sum_gradients.append([])
    cat_gradients.append([])
    losses = []
    for j in range(len(train_loaders[i])):
        if no_temporal and j == new_j:
            continue
        print('task %d for %s start training...' % (j,task_dirs[i]))
        running_loss = 0.0
        train_loader = train_loaders[i][j]
        val_loader = val_loaders[i][j]
        scaler = scalers[i][j]
        drop_out_p = 0.3

        model = build_baseline(args.baseline, device, num_nodes, in_dim, drop_out_p)
        model.to(device)
        learning_rate = 0.01
        weight_dacay = 0.001
        criterion = MaskedMAELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_dacay)
        for epoch in tqdm(range(first_stage_num_epochs), disable=True):

            train(scaler,train_loader,epoch,optimizer,criterion,model,device,in_dim,last_param=1,last_out=torch.zeros(output_shape))

            validate(scaler,val_loader,criterion,model,device,in_dim)
        sum_gradient, cat_gradient = compute_gradient(model,val_loader,device,criterion,in_dim,scaler)
        sum_gradients[i].append(sum_gradient)
        cat_gradients[i].append(cat_gradient)
n_clusters = 4

cluster_members = clustering_2d_list(n_clusters=n_clusters, tensor_2d_list=cat_gradients, algorithm='kmeans')
cluster_grad = get_cluster_average(cluster_members=cluster_members, cat_grad_dict=cat_gradients)

sum_gradients_cluster = [sum(cluster_grad[i]) for i in range(len(cluster_grad))]
sorted_clusters_gradients_list = sort_by_original_gradient(sum_gradients=sum_gradients_cluster)
sorted_sum_gradients = []
sorted_cat_gradients = []
train_cat_gradients = []
train_sum_gradients = []
second_stage_circle = 5
second_stage_num_epochs = 10
p_common = 0.5

common_model = build_baseline(args.baseline, device, num_nodes, in_dim, p_common)

weight_decay_common = 0.5
common_learning_rate = 0.01
common_model_optimizer = optim.Adam(common_model.parameters(), lr=common_learning_rate, weight_decay=weight_decay_common)
criterion = MaskedMAELoss()
p0=0.05
lambda0 = 0.05
common_train_loss = []
common_val_mae = []
common_val_rmse = []
common_val_mape = []
LTP_bool = True
LTD_bool = True
LTP = False
LTD = False

saved = False
train_ratio = 0.7
for circle in range(second_stage_circle):

    for cluster_index in sorted_clusters_gradients_list:

        cluster = cluster_members[cluster_index]
        temp_gradients_list = []
        for i in range(len(cluster)):
            temp_gradients_list.append(sum_gradients[cluster[i][0]][cluster[i][1]])

        temp_sorted_gradients_list = sort_by_original_gradient(sum_gradients=temp_gradients_list)
        sorted_gradients_list = [cluster[index] for index in temp_sorted_gradients_list]
        d_max = sum_gradients[sorted_gradients_list[-1][0]][sorted_gradients_list[-1][1]]
        last_param = 0
        last_output = torch.zeros(output_shape)
        random.shuffle(sorted_gradients_list)
        chosen_list = choose_sublist(sorted_gradients_list,train_ratio)
        for i_j_couple in chosen_list:
                i = i_j_couple[0]
                j = i_j_couple[1]
                if saved:
                    torch.save(common_model,f'./saved_models/{args.dataset}/common_model_{i}_{j}_start.pth')
                grad_norms = {name: [] for name, param in common_model.named_parameters()}
                sorted_sum_gradients.append(sum_gradients[i][j])
                sorted_cat_gradients.append(cat_gradients[i][j])
                if no_task and i == new_i:
                    continue
                if no_temporal and j == new_j:
                    continue

                print('task %d for %s start training...' % (j, task_dirs[i]))
                train_loader = train_loaders[i][j]
                val_loader = val_loaders[i][j]
                test_loader = test_loaders[i][j]

                scaler = scalers[i][j]

                difference_value = sum_gradients[i][j]
                p_common = p0

                weight_dacay_common = lambda0


                common_model.dropout = p_common
                common_model_optimizer = optim.Adam(common_model.parameters(), lr=common_learning_rate,
                                                    weight_decay=weight_decay_common)
                common_model.to(device)
                for epoch in tqdm(range(second_stage_num_epochs), disable=True):
                    sum_grad, cat_grad = compute_gradient(model=common_model, train_loader=train_loader, device=device,criterion=criterion,in_dim=in_dim,scaler=scaler)
                    los, activate_frequency = train(scaler,train_loader,epoch,common_model_optimizer,criterion,common_model,device,in_dim,
                                                    last_param=last_param,last_out=last_output)
                    train_cat_gradients.append(cat_grad)
                    train_sum_gradients.append(sum_grad)
                    common_train_loss.append(los)


                    mae,rmse,mape, out_put=validate(scaler, val_loader, criterion, common_model, device, in_dim)
                    common_val_mae.append(mae)
                    common_val_rmse.append(rmse)
                    common_val_mape.append(mape)
                last_param = mape
                last_output = out_put

                if saved:
                    draw_curve(grad_norms=grad_norms,save_dir=f'./gradient_visualization_{args.dataset}',key=[i,j],circle=circle,saved=True)
                    torch.save(common_model,f'./saved_models/{args.dataset}/common_model_{i}_{j}_end.pth')
coupled_train_loss = []
coupled_train_val_mape = []
coupled_train_val_mae=[]
coupled_train_val_rmse = []
fourth_stage_num_epochs = 100
print('task %d for %s start training...' % (new_j,task_dirs[new_i]))
coupled_model = common_model
coupled_model.to(device)
coupled_learning_rate = 0.01
coupled_weight_decay = 0.0001
coupled_optimizer = optim.Adam(coupled_model.parameters(), lr=coupled_learning_rate, weight_decay=coupled_weight_decay)
best_mae = 100
best_mape = 100
best_rmse = 100
for epoch in range(fourth_stage_num_epochs):


    los, _ = train(new_scaler,new_train_loader,epoch,coupled_optimizer,criterion,coupled_model,device,in_dim,
                                                    last_param=1,last_out=torch.zeros(output_shape))

    coupled_train_loss.append(los)
    mae, rmse, mape, _=validate(new_scaler,new_val_loader,criterion,coupled_model,device,in_dim)
    if mae < best_mae:
        best_mae = mae
    if mape < best_mape:
        best_mape = mape
    if rmse < best_rmse:
        best_rmse = rmse
    coupled_train_val_mae.append(mae)
    coupled_train_val_rmse.append(rmse)
    coupled_train_val_mape.append(mape)
print('best mae:',best_mae,'best rmse:',best_rmse,'best mape:',best_mape)
test_mae,test_rmse,test_mape=test(new_scaler,new_test_loader,criterion,coupled_model,device,in_dim)
if saved:
    torch.save(coupled_model,f'./saved_models/coupled_model_{args.dataset}.pth')
