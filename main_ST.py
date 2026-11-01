import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


from tools.metrics import *
from models.gwn import gwnet
from tools.dataloader import *
from tools.gradient_compute import *
import os
from torch.utils.data import Dataset
import torch.optim as optim
from engine import *
from tools.cluster_ST import *
from tools.util import *
import argparse
import matplotlib.pyplot as plt

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
args = parser.parse_args()
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
new_i = 1
if args.dataset == 'NYC': new_i=3
new_j = len(train_loaders[0])-1
few_shot_scale = 0.3
for i in range(len(task_dirs)):
    dataloaders, scaler, num_nodes = get_dataloaders_scaler_and_split_task(task_dirs[i], batch_size,task_per_dir=task_per_dir)
    dataloaders_list, scaler, num_nodes = get_dataloaders_scaler_and_split_task_few_shot(task_dirs[i], args.batch_size, 6, few_shot_scale, new_j)

    for j in range(len(dataloaders)):
        train_loaders[i][j]=(dataloaders[j]['train'])
        val_loaders[i][j]=(dataloaders[j]['val'])
        test_loaders[i][j]=(dataloaders[j]['test'])
    scalers.append(scaler) 
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
        model = gwnet(device=device, num_nodes=num_nodes, gcn_bool=True, addaptadj=True, in_dim=in_dim, dropout=drop_out_p)
        model.to(device)
        learning_rate = 0.01
        weight_dacay = 0.001
        criterion = MaskedMAELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_dacay)
        for epoch in tqdm(range(first_stage_num_epochs)):

            train(scaler,train_loader,epoch,optimizer,criterion,model,device,in_dim,last_param=1,last_out=torch.zeros(output_shape))

            validate(scaler,val_loader,criterion,model,device,in_dim)
        sum_gradient, cat_gradient = compute_gradient(model,val_loader,device,criterion,in_dim,scaler)
        sum_gradients[i].append(sum_gradient)
        cat_gradients[i].append(cat_gradient)
n_clusters = 2

cluster_members = clustering_2d_list(n_clusters=n_clusters, tensor_2d_list=cat_gradients, algorithm='kmeans')
cluster_grad = get_cluster_average(cluster_members=cluster_members, cat_grad_dict=cat_gradients)
sorted_clusters_gradients_list = find_path(cluster_grad)
sorted_sum_gradients = []
sorted_cat_gradients = []
train_cat_gradients = []
train_sum_gradients = []
second_stage_circle = 5
second_stage_num_epochs = 10
p_common = 0.5

common_model = gwnet(device=device,num_nodes=num_nodes,gcn_bool=True,addaptadj=True,in_dim=in_dim,dropout=p_common)

weight_decay_common = 0.5
common_learning_rate = 0.01
common_model_optimizer = optim.Adam(common_model.parameters(), lr=common_learning_rate, weight_decay=weight_decay_common)
criterion = MaskedMAELoss()
p0=0.05
zeta = 1.1
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
min_i = 0
min_j = 0
for circle in range(second_stage_circle):
    for cluster_index in sorted_clusters_gradients_list:

        cluster = cluster_members[cluster_index]
        temp_gradients_list = []
        for i in range(len(cluster)):
            temp_gradients_list.append(sum_gradients[cluster[i][0]][cluster[i][1]])
        temp_sorted_gradients_list = sort_by_original_gradient(sum_gradients=temp_gradients_list)
        sorted_gradients_list = [cluster[index] for index in temp_sorted_gradients_list]
        if cluster_index == 0:
             min_i = sorted_gradients_list[0][0]
             min_j = sorted_gradients_list[0][1]
        d_max = sum_gradients[sorted_gradients_list[-1][0]][sorted_gradients_list[-1][1]]
        last_param = 0
        last_output = torch.zeros(output_shape)
        for i_j_couple in sorted_gradients_list:
                i = i_j_couple[0]
                j = i_j_couple[1]
                if saved:
                    torch.save(common_model,f'./saved_models/{args.dataset}/common_model_{i}_{j}_start.pth')
                grad_norms = {name: [] for name, param in common_model.named_parameters()}
                if no_task and i == new_i:
                    continue
                if no_temporal and j == new_j:
                    continue
                sorted_sum_gradients.append(sum_gradients[i][j])
                sorted_cat_gradients.append(cat_gradients[i][j])

                print('task %d for %s start training...' % (j, task_dirs[i]))
                train_loader = train_loaders[i][j]
                val_loader = val_loaders[i][j]
                test_loader = test_loaders[i][j]

                scaler = scalers[i][j]

                difference_value = sum_gradients[i][j]
                p_common = my_sigmoid(p0, difference_value,d_max)
                weight_decay_common = my_sigmoid(lambda0,difference_value,d_max)


                common_model.dropout = p_common
                common_model_optimizer = optim.Adam(common_model.parameters(), lr=common_learning_rate,
                                                    weight_decay=weight_decay_common)
                common_model.to(device)
                for epoch in tqdm(range(second_stage_num_epochs)):
                    sum_grad, cat_grad = compute_gradient(model=common_model, train_loader=train_loader, device=device,criterion=criterion,in_dim=in_dim,scaler=scaler)
                    los, activate_frequency = train(scaler,train_loader,epoch,common_model_optimizer,criterion,common_model,device,in_dim,
                                                    last_param=last_param,last_out=last_output)
                    train_cat_gradients.append(cat_grad)
                    train_sum_gradients.append(sum_grad)
                    for name, param in common_model.named_parameters():
                        if param.grad is not None and 'weight' in name:
                            grad_norm = param.grad.norm().item()
                            grad_norms[name].append(grad_norm)
                    if LTP and LTP_bool and saved:
                                torch.save(common_model,f'./saved_models/LTP_post_common_model_{args.dataset}.pth')
                                LTP_bool = False
                    if LTD and LTD_bool and saved:
                                torch.save(common_model,f'./saved_models/LTD_post_common_model_{args.dataset}.pth')
                                LTD_bool = False
                    if epoch % 2 == 0:
                        if activate_frequency > 0.8:

                            if LTP_bool and saved:
                                torch.save(common_model,f'./saved_models/LTP_pre_common_model_{args.dataset}.pth')
                                
                                LTP = True
                                LTP_bool = False
                            weight_decay_common *= zeta
                            optimizer = torch.optim.Adam(common_model.parameters(), lr=common_learning_rate, weight_decay=weight_decay_common)
                            if common_model.dropout*zeta<1:
                                common_model.dropout *= zeta
                        elif activate_frequency<0.2:

                            if LTD_bool and saved:
                                torch.save(common_model,f'./saved_models/LTD_pre_common_model_{args.dataset}.pth')
                                
                                LTD = True
                                LTD_bool = False
                            weight_decay_common /=zeta
                            optimizer = torch.optim.Adam(common_model.parameters(), lr=common_learning_rate, weight_decay=weight_decay_common)
                            common_model.dropout/=zeta
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
third_stage_circle = 5
third_stage_num_epochs = 20
p_personal = 0.1
personal_extractor = gwnet(device=device,num_nodes=num_nodes,gcn_bool=True,addaptadj=True,in_dim=in_dim,dropout=p_personal)
weight_decay_personal = 0.1
personal_learning_rate = 0.01
personal_extractor_optimizer = optim.Adam(personal_extractor.parameters(), lr=personal_learning_rate, weight_decay=weight_decay_personal)
base_train_loader = train_loaders[min_i][min_j]
base_scaler = scalers[min_i][min_j]
local_min_i = min_i
local_min_j = min_j
personal_extractor.to(device)
for _ in range(third_stage_circle):
    for i in range(len(task_dirs)):
        if no_task and i == new_i:
            continue
        for j in range(len(task_dirs[i])):
            if no_temporal and j == new_j: continue
            contrastive_train_loader = train_loaders[i][j]
            contrastive_val_loader = val_loaders[i][j]
            contrastive_test_loader = test_loaders[i][j]
            contrastive_scaler = scalers[i][j]
            for epoch in range(third_stage_num_epochs):
                contrastive_train(base_train_loader,contrastive_train_loader,personal_extractor,base_scaler,contrastive_scaler
                            ,in_dim,device,epoch,criterion,personal_extractor_optimizer,batch_size)
            local_min_i = i
            local_min_j = j
            base_scaler = scalers[local_min_i][local_min_j]
            base_train_loader = train_loaders[local_min_i][local_min_j]
        contrastive_train_loader = train_loaders[min_i][min_j]
        contrastive_val_loader = val_loaders[min_i][min_j]
        contrastive_test_loader = test_loaders[min_i][min_j]
        contrastive_scaler = scalers[min_i][min_j]
        for epoch in range(third_stage_num_epochs):
            contrastive_train(base_train_loader, contrastive_train_loader, personal_extractor, base_scaler, contrastive_scaler
                        , in_dim, device, epoch, criterion, personal_extractor_optimizer, batch_size)
coupled_train_loss = []
coupled_train_val_mape = []
coupled_train_val_mae=[]
coupled_train_val_rmse = []
fourth_stage_num_epochs = 30
print('task %d for %s start training...' % (new_j,task_dirs[new_i]))
coupled_model = common_model
coupled_model.to(device)
coupled_learning_rate = 0.01
coupled_weight_decay = 0.0001
coupled_optimizer = optim.Adam(coupled_model.parameters(), lr=coupled_learning_rate, weight_decay=coupled_weight_decay)
kappa=5
G_store = get_G_store(personal_extractor=personal_extractor,train_loaders=train_loaders)
new_re = get_new_re(new_train_loader=new_train_loader,personal_extractor=personal_extractor)
min_distance = min_euclidean_distance(query_vector=new_re,vector_store=G_store)
best_mae = 100
best_mape = 100
best_rmse = 100
if min_distance<kappa:
    coupled_model =common_model
    coupled_model.to(device)
    coupled_model.dropout = my_sigmoid(p0,difference_value)
    coupled_weight_decay = my_sigmoid(lambda0,difference_value)
    coupled_optimizer = optim.Adam(coupled_model.parameters(), lr=coupled_learning_rate, weight_decay=coupled_weight_decay)
    for epoch in range(fourth_stage_num_epochs):
        coupled_train_loss.append(
            train(new_scaler, new_train_loader, epoch, coupled_optimizer, criterion, coupled_model, device, in_dim))
        mae, rmse, mape = validate(new_scaler, new_val_loader, criterion, coupled_model, device, in_dim)
        coupled_train_val_mae.append(mae)
        coupled_train_val_rmse.append(rmse)
        coupled_train_val_mape.append(mape)
else:
    coupled_model = gwnet(device=device,num_nodes=num_nodes,gcn_bool=True,addaptadj=True,in_dim=in_dim,dropout=0.1*p0)
    coupled_optimizer = optim.Adam(coupled_model.parameters(), lr=coupled_learning_rate, weight_decay=0.1*lambda0)
    for epoch in range(fourth_stage_num_epochs):
        coupled_train_loss.append(
            train(new_scaler, new_train_loader, epoch, coupled_optimizer, criterion, coupled_model, device, in_dim))
        mae, rmse, mape = validate(new_scaler, new_val_loader, criterion, coupled_model, device, in_dim)
        coupled_train_val_mae.append(mae)
        coupled_train_val_rmse.append(rmse)
        coupled_train_val_mape.append(mape)
test_mae, test_rmse, test_mape = test(new_scaler, new_test_loader, criterion, coupled_model, device, in_dim)
print('best mae:',best_mae,'best rmse:',best_rmse,'best mape:',best_mape)
