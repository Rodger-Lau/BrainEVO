import math

import torch
import numpy as np
import os

class StandardScaler:
    
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit_transform(self, data):
        self.mean = data.mean()
        self.std = data.std()

        return (data - self.mean) / self.std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean
def get_dataloaders_scaler_and_split_task(dataset_dir, batch_size=16, task_per_dir=4, logger=None):
    print("=== get_dataloaders_scaler_and_split_task START ===")
    data = {}

    for category in ['train', 'val', 'test']:
        file_path = os.path.join(dataset_dir, category + '.npz')
        print(f"Loading {category} data from {file_path} ...")
        cat_data = np.load(file_path)
        
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y'][..., :1]

        print(f"{category} data shape: x_{category}={data['x_'+category].shape}, "
              f"y_{category}={data['y_'+category].shape}")


    num_nodes = data['x_train'].shape[2]
    print(f"Number of nodes: {num_nodes}")


    print("Extracting hours from x_train/x_val/x_test...")
    hour_train = data['x_train'][:, 11, 0, 8].astype(int)
    hour_val   = data['x_val'][:,   11, 0, 8].astype(int)
    hour_test  = data['x_test'][:,  11, 0, 8].astype(int)

    print(f"hour_train shape = {hour_train.shape}, hour_val shape = {hour_val.shape}, hour_test shape = {hour_test.shape}")


    print("Calculating group IDs for each sample...")
    group_train = hour_train // (24 // task_per_dir)
    group_val   = hour_val   // (24 // task_per_dir)
    group_test  = hour_test  // (24 // task_per_dir)


    train_tasks = [np.where(group_train == i)[0] for i in range(task_per_dir)]
    val_tasks   = [np.where(group_val   == i)[0] for i in range(task_per_dir)]
    test_tasks  = [np.where(group_test  == i)[0] for i in range(task_per_dir)]

    for i in range(task_per_dir):
        print(f"Group {i} -> train_samples: {len(train_tasks[i])}, "
              f"val_samples: {len(val_tasks[i])}, test_samples: {len(test_tasks[i])}")

    dataloaders = []
    scalers = []


    print("Creating DataLoaders for each group/task...")
    for i in range(task_per_dir):
        print(f"=== Now processing group {i} ===")
        sel_train_idx = train_tasks[i]
        sel_val_idx   = val_tasks[i]
        sel_test_idx  = test_tasks[i]


        if len(sel_train_idx) == 0:
            msg = f"[Warning] No training samples in group {i}."
            print(msg)
            if logger is not None:
                logger.warning(msg)
        mean_val = data['x_train'][sel_train_idx, ..., 0].mean() if len(sel_train_idx) > 0 else 0
        std_val  = data['x_train'][sel_train_idx, ..., 0].std()  if len(sel_train_idx) > 0 else 1e-8
        scaler = StandardScaler(mean=mean_val, std=std_val)

        print(f"Scaler for group {i}: mean={scaler.mean}, std={scaler.std}")


        data['x_train'][sel_train_idx, ..., 0] = scaler.transform(data['x_train'][sel_train_idx, ..., 0])
        data['x_val'][sel_val_idx,     ..., 0] = scaler.transform(data['x_val'][sel_val_idx,     ..., 0])
        data['x_test'][sel_test_idx,   ..., 0] = scaler.transform(data['x_test'][sel_test_idx,   ..., 0])


        print(f"Constructing TensorDataset for group {i}...")
        dataset_train = torch.utils.data.TensorDataset(
            torch.FloatTensor(data['x_train'][sel_train_idx]),
            torch.FloatTensor(data['y_train'][sel_train_idx])
        )
        dataset_val = torch.utils.data.TensorDataset(
            torch.FloatTensor(data['x_val'][sel_val_idx]),
            torch.FloatTensor(data['y_val'][sel_val_idx])
        )
        dataset_test = torch.utils.data.TensorDataset(
            torch.FloatTensor(data['x_test'][sel_test_idx]),
            torch.FloatTensor(data['y_test'][sel_test_idx])
        )


        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        dataloader_val   = torch.utils.data.DataLoader(dataset_val,   batch_size=batch_size, shuffle=False)
        dataloader_test  = torch.utils.data.DataLoader(dataset_test,  batch_size=batch_size, shuffle=False)

        dataloader_dict = {
            "train": dataloader_train,
            "val":   dataloader_val,
            "test":  dataloader_test
        }

        dataloaders.append(dataloader_dict)
        scalers.append(scaler)
        print(f"Group {i} DataLoader created: "
              f"train_size={len(dataset_train)}, val_size={len(dataset_val)}, test_size={len(dataset_test)}")

    print("=== get_dataloaders_scaler_and_split_task DONE ===")
    return dataloaders, scalers, num_nodes

def get_dataloaders_scaler_and_split_task_few_shot(dataset_dir, batch_size=16, task_per_dir=4, few_shot_scale=1, new_j=0):
    print("=== get_dataloaders_scaler_and_split_task START ===")
    data = {}

    for category in ['train', 'val', 'test']:
        file_path = os.path.join(dataset_dir, category + '.npz')
        print(f"Loading {category} data from {file_path} ...")
        cat_data = np.load(file_path)
        
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y'][..., :1]

        print(f"{category} data shape: x_{category}={data['x_'+category].shape}, "
              f"y_{category}={data['y_'+category].shape}")


    num_nodes = data['x_train'].shape[2]
    print(f"Number of nodes: {num_nodes}")


    print("Extracting hours from x_train/x_val/x_test...")
    hour_train = data['x_train'][:, 11, 0, 8].astype(int)
    hour_val   = data['x_val'][:,   11, 0, 8].astype(int)
    hour_test  = data['x_test'][:,  11, 0, 8].astype(int)

    print(f"hour_train shape = {hour_train.shape}, hour_val shape = {hour_val.shape}, hour_test shape = {hour_test.shape}")


    print("Calculating group IDs for each sample...")
    group_train = hour_train // (24 // task_per_dir)
    group_val   = hour_val   // (24 // task_per_dir)
    group_test  = hour_test  // (24 // task_per_dir)


    train_tasks = [np.where(group_train == j)[0] for j in range(task_per_dir)]
    val_tasks   = [np.where(group_val   == j)[0] for j in range(task_per_dir)]
    test_tasks  = [np.where(group_test  == j)[0] for j in range(task_per_dir)]

    for j in range(task_per_dir):
        print(f"Group {j} -> train_samples: {len(train_tasks[j])}, "
              f"val_samples: {len(val_tasks[j])}, test_samples: {len(test_tasks[j])}")

    dataloaders = []
    scalers = []
    cur_means = []
    cur_stds = []

    print("Creating DataLoaders for each group/task...")
    for i in range(task_per_dir):
        print(f"=== Now processing group {i} ===")
        sel_train_idx = train_tasks[i]

        if i == new_j:
            few_train_num = int(len(sel_train_idx) * few_shot_scale)
            sel_train_idx = np.random.choice(sel_train_idx, size=few_train_num, replace=False)
        print(f'sel_train_idx: {sel_train_idx}, {type(sel_train_idx)}')
        sel_val_idx   = val_tasks[i]
        sel_test_idx  = test_tasks[i]


        if len(sel_train_idx) == 0:
            msg = f"[Warning] No training samples in group {i}."
            print(msg)


        mean_val = data['x_train'][sel_train_idx, ..., 0].mean() if len(sel_train_idx) > 0 else 0
        std_val  = data['x_train'][sel_train_idx, ..., 0].std()  if len(sel_train_idx) > 0 else 1e-8
        cur_mean = np.mean(data['x_train'][sel_train_idx], axis=(0,1,2)) if len(sel_train_idx) > 0 else np.zeros(len(data['x_train'][sel_train_idx][0][0]))
        cur_std = np.std(data['x_train'][sel_train_idx], axis=(0,1,2)) if len(sel_train_idx) > 0 else np.zeros(len(data['x_train'][sel_train_idx][0][0]))+(1e-8)
        scaler = StandardScaler(mean=mean_val, std=std_val)

        print(f"Scaler for group {i}: mean={scaler.mean}, std={scaler.std}")


        data['x_train'][sel_train_idx, ..., 0] = scaler.transform(data['x_train'][sel_train_idx, ..., 0])
        data['x_val'][sel_val_idx,     ..., 0] = scaler.transform(data['x_val'][sel_val_idx,     ..., 0])
        data['x_test'][sel_test_idx,   ..., 0] = scaler.transform(data['x_test'][sel_test_idx,   ..., 0])


        print(f"Constructing TensorDataset for group {i}...")
        dataset_train = torch.utils.data.TensorDataset(
            torch.FloatTensor(data['x_train'][sel_train_idx]),
            torch.FloatTensor(data['y_train'][sel_train_idx])
        )
        dataset_val = torch.utils.data.TensorDataset(
            torch.FloatTensor(data['x_val'][sel_val_idx]),
            torch.FloatTensor(data['y_val'][sel_val_idx])
        )
        dataset_test = torch.utils.data.TensorDataset(
            torch.FloatTensor(data['x_test'][sel_test_idx]),
            torch.FloatTensor(data['y_test'][sel_test_idx])
        )


        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        dataloader_val   = torch.utils.data.DataLoader(dataset_val,   batch_size=batch_size, shuffle=False)
        dataloader_test  = torch.utils.data.DataLoader(dataset_test,  batch_size=batch_size, shuffle=False)

        dataloader_dict = {
            "train": dataloader_train,
            "val":   dataloader_val,
            "test":  dataloader_test
        }

        dataloaders.append(dataloader_dict)
        scalers.append(scaler)
        cur_means.append(cur_mean)
        cur_stds.append(cur_std)
        print(f"Group {i} DataLoader created: "
              f"train_size={len(dataset_train)}, val_size={len(dataset_val)}, test_size={len(dataset_test)}")

    print("=== get_dataloaders_scaler_and_split_task DONE ===")
    return dataloaders, scalers, num_nodes