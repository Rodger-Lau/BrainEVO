import math

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F


def sort_by_original_gradient(sum_gradients):

    if not isinstance(sum_gradients, list):
        return [0]
    

    if len(sum_gradients) == 0:
        return []
    sorted_index_list = [i for i, _ in sorted(enumerate(sum_gradients), key=lambda x: x[1])]
    return sorted_index_list
def tensor_difference(a, b):
    return torch.sqrt(torch.sum(torch.pow(a-b,2)))
def sorted_indices_2d(matrix):
    pass
    if not matrix or not any(matrix):
        return []
    

    elements = []
    for row_idx, row in enumerate(matrix):
        for col_idx, value in enumerate(row):
            elements.append((value, row_idx, col_idx))
    

    sorted_elements = sorted(elements, key=lambda x: x[0])
    

    return [(row, col) for _, row, col in sorted_elements]

def compute_gradient(model,train_loader,device,criterion,in_dim,scaler):
    grad_sum = 0
    grad_cat = []
    for train_data in train_loader:
        inputs, reals = train_data


        inputs = torch.Tensor(inputs[:, :, :, :in_dim])
        reals = torch.Tensor(reals)

        reals = reals.to(device)
        inputs = inputs.transpose(1, 3)
        inputs = nn.functional.pad(inputs, (1, 0, 0, 0))
        inputs = inputs.to(device)

        outputs = model(inputs)[1]
        predict = scaler.inverse_transform(outputs)
        loss = criterion(predict, reals)
        loss /= loss.item()


        loss.backward()
        parameterss = []
        grads = []

        for param in model.parameters():
            if param.grad is not None:
                parameterss.append(torch.sum(torch.pow(param.grad,2)))
                grads.append(param.grad.view(-1))
            grad_sum = sum(parameterss)
            grad_cat = torch.cat(grads)
        return grad_sum,grad_cat.cpu()


def get_min_gradient(sum_gradients,new_i,new_j):
    min = sum_gradients[0][0]
    min_i = 0
    min_j = 0
    for i in range(len(sum_gradients)):
        for j in range(len(sum_gradients[i])):
            if i==new_i and j==new_j:continue
            if sum_gradients[i][j]<min:
                min_i = i
                min_j = j
                min = sum_gradients[i][j]
    return min, min_i, min_j
def reverse_sort_sample_by_gradient(min_i,min_j,cat_gradients):
    sample_order = []
    min_sample = cat_gradients[min_i][min_j]
    difference_matrix = []
    difference_matrix_copy = []
    row = len(cat_gradients)
    column = len(cat_gradients[0])

    for i in range(len(cat_gradients)):
        difference_matrix.append([])
        difference_matrix_copy.append([])
        for j in range(len(cat_gradients[i])):
            difference_matrix[i].append(tensor_difference(min_sample, cat_gradients[i][j]))
            difference_matrix_copy[i].append(tensor_difference(min_sample, cat_gradients[i][j]))
    max_value = np.max(difference_matrix_copy)
    for _ in range(row * column):
        index = np.argmax(difference_matrix)
        i = index // column
        j = index % column
        sample_order.append([i, j])
        difference_matrix[i][j] = -1e+200
    return sample_order, difference_matrix_copy, max_value
def sort_sample_by_gradient(min_i,min_j,cat_gradients):
    sample_order = []
    min_sample = cat_gradients[min_i][min_j]
    difference_matrix = []
    difference_matrix_copy = []
    row = len(cat_gradients)
    column = len(cat_gradients[0])

    for i in range(len(cat_gradients)):
        difference_matrix.append([])
        difference_matrix_copy.append([])
        for j in range(len(cat_gradients[i])):
            difference_matrix[i].append(tensor_difference(min_sample, cat_gradients[i][j]))
            difference_matrix_copy[i].append(tensor_difference(min_sample, cat_gradients[i][j]))
    max_value = np.max(difference_matrix_copy)
    for _ in range(row*column):
        index = np.argmin(difference_matrix)
        i = index // column
        j = index % column
        sample_order.append([i,j])
        difference_matrix[i][j] = 1e+200
    return sample_order,difference_matrix_copy,max_value


def my_sigmoid(p0,x,d_max,circle=2):
    return p0*(1-math.exp(x-d_max))


def my_reverse(p0,x):
    return p0*(1/(x+1))
def dynamic_sigmoid(x):
    return 1/math.exp(-1*x)

def dynamic_coupling(common_model,personal_extractor,device,num_nodes,new_sample,criterion,in_dim,scaler):

    coupled_model = common_model

    sum_gradient, cat_gradient = compute_gradient(common_model, new_sample, device, criterion, in_dim, scaler)

    lambda1 = dynamic_sigmoid(-sum_gradient)
    for (coupled_param,common_param,personal_param) in zip(coupled_model.named_parameters(),common_model.named_parameters(),personal_extractor.named_parameters()):

        if 'end_conv_2' not in coupled_param[0] and 'end_conv_3' not in coupled_param[0]:
            coupled_param[1].data = (lambda1*common_param[1]+(1-lambda1)*personal_param[1]).data
    coupled_model.dropout = 0.7
    return coupled_model

