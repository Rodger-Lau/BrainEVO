import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
#import tools.my_utils
#from tools.my_utils import *
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
    plt.yscale('log')  # 对数坐标更清晰
    plt.title("Gradient Norms per Layer")
    plt.legend()
    #plt.show()
    if saved:
        plt.savefig(os.path.join(save_dir, f'circle{circle}_{key[0]}_{key[1]}_gradient.pdf'))
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default = None)
args = parser.parse_args()
dataset_path = args.dataset
# dataset_path = 'CHI'
if dataset_path == 'CHI':
    tasks = ['RISK','TAXIDROP','TAXIPICK']
elif dataset_path == 'NYC':
    tasks = ['CROWDIN','CROWDOUT','TAXIDROP','TAXIPICK']
else:
    tasks = ['FLOW','SPEED']
#dataset_path = './data/data/CHI'
dataset_path = os.path.join('./data/data',dataset_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#tasks = os.listdir(dataset_path)
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
#saved = False
# 划分tasks
first_stage_num_epochs = 3
new_i = 1
if args.dataset == 'NYC': new_i=3
new_j = len(train_loaders[0])-1
few_shot_scale = 0.3
for i in range(len(task_dirs)):
    dataloaders, scaler, num_nodes = get_dataloaders_scaler_and_split_task(task_dirs[i], batch_size,task_per_dir=task_per_dir)
    dataloaders_list, scaler, num_nodes = get_dataloaders_scaler_and_split_task_few_shot(task_dirs[i], args.batch_size, 6, few_shot_scale, new_j)
    #print(len(dataloaders))
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
no_task = False
no_temporal = False
sum_gradients = []
cat_gradients = []
print('we skip task %d for %s'%(new_j,task_dirs[new_i]))
# 计算梯度并保存
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
            #print(len(train_loader))
            train(scaler,train_loader,epoch,optimizer,criterion,model,device,in_dim,last_param=1,last_out=torch.zeros(output_shape))
            #if epoch % 10 == 0:
            validate(scaler,val_loader,criterion,model,device,in_dim)
        sum_gradient, cat_gradient = compute_gradient(model,val_loader,device,criterion,in_dim,scaler)
        print(sum_gradient,len(cat_gradient))
        sum_gradients[i].append(sum_gradient)
        cat_gradients[i].append(cat_gradient)
n_clusters = 2

cluster_members = clustering_2d_list(n_clusters=n_clusters, tensor_2d_list=cat_gradients, algorithm='kmeans')
print(cluster_members)
cluster_grad = get_cluster_average(cluster_members=cluster_members, cat_grad_dict=cat_gradients)
print(cluster_grad)
#min_grad, min_i,min_j = get_min_gradient(sum_gradients=cluster_grad, new_i=-1, new_j=-1)
# sum_gradients_cluster = [sum(cluster_grad[i]) for i in range(len(cluster_grad))]
# sorted_clusters_gradients_list = sort_by_original_gradient(sum_gradients=sum_gradients_cluster)
sorted_clusters_gradients_list = find_path(cluster_grad)
# sum_gradients_cluster=[]
# for i in range(len(cluster_grad)):
#     sum_gradients_cluster.append([sum(cluster_grad[j]) for j in range(len(cluster_grad[i]))])
print(sorted_clusters_gradients_list)
# min_gradient,min_i,min_j = get_min_gradient(sum_gradients,new_i,new_j)
# sample_order,difference_matrix,d_max = sort_sample_by_gradient(min_i,min_j,cat_gradients)  # sample_order中按顺序保存着[i，j]
# print(d_max)
# print(sample_order)


# 构造common_model
sorted_sum_gradients = []
sorted_cat_gradients = []
train_cat_gradients = []
train_sum_gradients = []
second_stage_circle = 5
second_stage_num_epochs = 10
p_common = 0.5
#common_model = gwnet(device=device,num_nodes=num_nodes,gcn_bool=True,addaptadj=True,in_dim=in_dim,dropout=p_common,supports=adj_matrix)
common_model = gwnet(device=device,num_nodes=num_nodes,gcn_bool=True,addaptadj=True,in_dim=in_dim,dropout=p_common)
#personal_extractor = gwnet(device=device,num_nodes=num_nodes,gcn_bool=True,addaptadj=True,in_dim=in_dim,dropout=0.3)
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
# no_LTP_LTD=False
saved = False
min_i = 0
min_j = 0
for circle in range(second_stage_circle):
    for cluster_index in sorted_clusters_gradients_list:
        #print(cluster_index)
        cluster = cluster_members[cluster_index]  # cluster中含有domain
        temp_gradients_list = []
        for i in range(len(cluster)):
            temp_gradients_list.append(sum_gradients[cluster[i][0]][cluster[i][1]])
        temp_sorted_gradients_list = sort_by_original_gradient(sum_gradients=temp_gradients_list)
        sorted_gradients_list = [cluster[index] for index in temp_sorted_gradients_list]
        print(sorted_gradients_list)
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
                #if i==new_i:continue
                print('task %d for %s start training...' % (j, task_dirs[i]))
                train_loader = train_loaders[i][j]
                val_loader = val_loaders[i][j]
                test_loader = test_loaders[i][j]

                scaler = scalers[i][j]
                #difference_value = difference_matrix[i][j]  # difference value越小越靠前，且值>=0
                difference_value = sum_gradients[i][j]
                #d_max = sorted_gradients_list[-1]
                #if not (i == min_i and j == min_j):
                #p_common = my_sigmoid(-difference_value,circle+1)
                p_common = my_sigmoid(p0, difference_value, d_max)
                weight_decay_common = my_sigmoid(lambda0, difference_value, d_max)
                # p_common = my_reverse(p0,difference_value)
                # weight_decay_common = 0.1*my_reverse(p0,difference_value)
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
                                print('发生LTP，保存post_LTP')
                                LTP_bool = False
                    if LTD and LTD_bool and saved:
                                torch.save(common_model,f'./saved_models/LTD_post_common_model_{args.dataset}.pth')
                                print('发生LTD，保存post_LTD')
                                LTD_bool = False
                    if epoch % 2 == 0:
                        print('第',epoch, '轮激活频率为', activate_frequency)
                        if activate_frequency > 0.2:
                            #print('增强')
                            if LTP_bool and saved:
                                torch.save(common_model,f'./saved_models/LTP_pre_common_model_{args.dataset}.pth')
                                
                                LTP = True
                                LTP_bool = False
                            weight_decay_common *= 1.2
                            optimizer = torch.optim.Adam(common_model.parameters(), lr=common_learning_rate, weight_decay=weight_decay_common)
                            if common_model.dropout*1.2<1:
                                common_model.dropout *= 1.2
                        elif activate_frequency<0.1:
                            #print('削弱')
                            if LTD_bool and saved:
                                torch.save(common_model,f'./saved_models/LTD_pre_common_model_{args.dataset}.pth')
                                
                                LTD = True
                                LTD_bool = False
                            weight_decay_common /=1.2
                            optimizer = torch.optim.Adam(common_model.parameters(), lr=common_learning_rate, weight_decay=weight_decay_common)
                            common_model.dropout/=1.2
                    common_train_loss.append(los)
                    #common_train_loss.append(train(scaler,train_loader,epoch,common_model_optimizer,criterion,common_model,device,in_dim))
                #if epoch % 10 == 0:
                    mae,rmse,mape, out_put=validate(scaler, val_loader, criterion, common_model, device, in_dim)
                    common_val_mae.append(mae)
                    common_val_rmse.append(rmse)
                    common_val_mape.append(mape)
                last_param = mape
                last_output = out_put
                #draw_curve(grad_norms=grad_norms,save_dir=f'./gradient_visualization_{args.dataset}',key=[i,j],circle=circle,saved=True)
                if saved:
                    draw_curve(grad_norms=grad_norms,save_dir=f'./gradient_visualization_{args.dataset}',key=[i,j],circle=circle,saved=True)
                    torch.save(common_model,f'./saved_models/{args.dataset}/common_model_{i}_{j}_end.pth')
                    print(f'已保存{args.dataset}_{i}_{j}')
# if saved:
#     torch.save(common_model.state_dict(),'./common_model_CHI.pth')
#
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
        # i = i_j_couple[0]
        # j = i_j_couple[1]
            print('contrastive training between \'task %d for %s\' and \'task %d for %s\' starts......' % (local_min_j, task_dirs[local_min_i], j, task_dirs[i]))
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
        print('contrastive training between \'task %d for %s\' and \'task %d for %s\' starts......' % (local_min_j, task_dirs[local_min_i], min_j, task_dirs[min_i]))
        #personal_extractor.to(device)
        contrastive_train_loader = train_loaders[min_i][min_j]
        contrastive_val_loader = val_loaders[min_i][min_j]
        contrastive_test_loader = test_loaders[min_i][min_j]
        contrastive_scaler = scalers[min_i][min_j]
        for epoch in range(third_stage_num_epochs):
            contrastive_train(base_train_loader, contrastive_train_loader, personal_extractor, base_scaler, contrastive_scaler
                        , in_dim, device, epoch, criterion, personal_extractor_optimizer, batch_size)
# if saved:
#     torch.save(personal_extractor.state_dict(),'./personal_extractor_CHI_2.pth')
#
coupled_train_loss = []
coupled_train_val_mape = []
coupled_train_val_mae=[]
coupled_train_val_rmse = []
fourth_stage_num_epochs = 30
print('task %d for %s start training...' % (new_j,task_dirs[new_i]))
# #
# new_sample = test_loaders[min_i][min_j]
# new_sample_scaler = get_scaler(new_sample)
# val_loader = val_loaders[min_i][min_j]
# #coupled_model = dynamic_coupling(common_model,personal_extractor,device,num_nodes,new_sample,criterion,in_dim,new_sample_scaler)
#
# #coupled_model = dynamic_coupling(common_model,personal_extractor,device,num_nodes,new_train_loader,criterion,in_dim,new_scaler)
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
# if saved:
#     torch.save(coupled_model,f'./saved_models/coupled_model_{args.dataset}.pth')

