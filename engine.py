import copy
import random

from tools.metrics import *
import torch
import torch.nn as nn
class MaskedMAELoss(torch.nn.Module):

    def __init__(self):
        super(MaskedMAELoss, self).__init__()

    def forward(self, v_, v):
        mask = (v != 0.0)
        mask = mask.float()
        mask /= torch.mean((mask))
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
        loss = torch.abs(v_ - v)
        loss = loss * mask
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        return torch.mean(loss)

class MaskedMSELoss(torch.nn.Module):

    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, v_, v):
        mask = (v != 0.0)
        mask = mask.float()
        mask /= torch.mean((mask))
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
        loss = (v_ - v)**2
        loss = loss * mask
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        return torch.mean(loss)

class MaskedMAPELoss(torch.nn.Module):

    def __init__(self):
        super(MaskedMAPELoss, self).__init__()

    def forward(self, v_, v):
        mask = (v != 0.0)
        mask = mask.float()
        mask /= torch.mean((mask))
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
        loss = torch.abs(v_ - v)/v
        loss = loss * mask
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        return torch.mean(loss)

def contrastive_train(base_train_loader,contrastive_train_loader,personal_extractor,base_scaler,contrastive_scaler,in_dim,device,epoch,criterion,optimizer,batch_size,lambda1=1,lambda2=1):

    running_loss = 0.0
    count = 0
    for (base_data, contrastive_data) in zip(base_train_loader,contrastive_train_loader):
        base_inputs, base_reals = base_data

        contrastive_inputs, contrastive_reals = contrastive_data
        if len(base_inputs) == len(contrastive_inputs) and len(base_inputs) == batch_size:

            base_inputs = torch.Tensor(base_inputs[:, :, :, :in_dim])
            contrastive_inputs = torch.Tensor(contrastive_inputs[:, :, :, :in_dim])
            base_inputs = base_inputs.transpose(1, 3)
            base_inputs = nn.functional.pad(base_inputs, (1, 0, 0, 0))
            base_inputs = base_inputs.to(device)
            contrastive_inputs = contrastive_inputs.transpose(1, 3)
            contrastive_inputs = nn.functional.pad(contrastive_inputs, (1, 0, 0, 0))
            contrastive_inputs = contrastive_inputs.to(device)
            optimizer.zero_grad()
            base_outputs = personal_extractor(base_inputs)[0]


            if count>0:


                optimizer.zero_grad()
                loss = max(torch.tensor(2,dtype=torch.float,requires_grad=True),criterion(last_base_outputs,base_outputs))

                running_loss+=loss.item()
                if loss>1:
                    loss.backward()
                    optimizer.step()
            contrastive_outputs = personal_extractor(contrastive_inputs)[0]

            if count>0:
                optimizer.zero_grad()
                loss = max(torch.tensor(2,dtype=torch.float,requires_grad=True),criterion(last_contrastive_outputs,contrastive_outputs))

                running_loss+=loss.item()
                if loss>1:
                    loss.backward()
                    optimizer.step()
            base_outputs = personal_extractor(base_inputs)[0]

            contrastive_outputs = personal_extractor(contrastive_inputs)[0]

            optimizer.zero_grad()
            loss = 1e4/criterion(base_outputs,contrastive_outputs)
            running_loss-=criterion(base_outputs,contrastive_outputs).item()
            loss.backward()
            optimizer.step()
            count = 1
            last_base_outputs = copy.deepcopy(base_outputs.data)
            last_contrastive_outputs = copy.deepcopy(contrastive_outputs.data)
    print('Epoch %d loss: %.3f' % (epoch + 1, running_loss / len(base_train_loader)))


def train(scaler,train_loader,epoch,optimizer,criterion,model,device,in_dim,last_param=1,last_out=torch.zeros([1,1]),mode='mape'):
    last_out = last_out.to(device)

    if mode == 'mape':
        last_param = 1-last_param
    last_param *= 0.01
    total_activate_num = 0
    total_forward_num = 0

    running_loss = 0.0
    for train_data in train_loader:
        total_forward_num += 1
        model.train()
        inputs, reals = train_data


        inputs = torch.Tensor(inputs[:, :, :, :in_dim])
        reals = torch.Tensor(reals)

        reals = reals.to(device)
        inputs = inputs.transpose(1, 3)
        inputs = nn.functional.pad(inputs, (1, 0, 0, 0))
        inputs = inputs.to(device)
        optimizer.zero_grad()
        _, outputs, activate_num = model(inputs)
        total_activate_num += activate_num
        predict = scaler.inverse_transform(outputs)

        if last_out.shape!=reals.shape:
            last_out = reals

        target = last_param * last_out + (1 - last_param) * reals

        loss = criterion(predict,target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print('Epoch %d loss: %.3f' % (epoch + 1, running_loss / len(train_loader)))
    return running_loss / len(train_loader), total_activate_num / total_forward_num


def validate(scaler,val_loader,criterion,model,device,in_dim):
    model.eval()
    running_loss = 0.0
    running_mae = 0.0
    running_rmse = 0.0
    running_mape = 0.0
    with torch.no_grad():
        for val_data in val_loader:
            inputs, reals = val_data


            inputs = torch.Tensor(inputs[:, :, :, :in_dim])
            reals = torch.Tensor(reals)

            reals = reals.to(device)
            inputs = inputs.transpose(1, 3)
            inputs = nn.functional.pad(inputs, (1, 0, 0, 0))
            inputs = inputs.to(device)
            _,outputs,_ = model(inputs)
            predict = scaler.inverse_transform(outputs)

            mae,rmse,mape = get_mae_rmse_mape(predict, reals)

            running_mae+=mae
            running_mape+=mape
            running_rmse+=rmse
    print('val mae:%.3f, val rmse:%.3f, val mape:%.3f' % (running_mae / len(val_loader),running_rmse/len(val_loader),running_mape/len(val_loader)))
    return running_mae / len(val_loader),running_rmse/len(val_loader),running_mape/len(val_loader), outputs

def test(scaler, test_loader, criterion, model, device, in_dim):
    model.eval()
    running_loss = 0.0
    running_mae = 0.0
    running_rmse = 0.0
    running_mape = 0.0
    with torch.no_grad():
        for test_data in test_loader:
            inputs, reals = test_data


            inputs = torch.Tensor(inputs[:, :, :, :in_dim])
            reals = torch.Tensor(reals)

            reals = reals.to(device)
            inputs = inputs.transpose(1, 3)
            inputs = nn.functional.pad(inputs, (1, 0, 0, 0))
            inputs = inputs.to(device)
            outputs = model(inputs)[1]
            predict = scaler.inverse_transform(outputs)

            mae, rmse, mape = get_mae_rmse_mape(predict, reals)

            running_mae += mae
            running_mape += mape
            running_rmse += rmse
    print('test mae:%.3f, test rmse:%.3f, test mape:%.3f' % (
        running_mae / len(test_loader), running_rmse / len(test_loader), running_mape / len(test_loader)))
    return running_mae / len(test_loader), running_rmse / len(test_loader), running_mape / len(test_loader)


def coupled_train(scaler, train_loader, epoch, optimizer_common, optimizer_personal, criterion, common_model, personal_extractor, adaptive_common_layer, adaptive_personal_layer, device, in_dim):

    running_loss = 0.0
    for train_data in train_loader:
        adaptive_common_layer.train()
        inputs, reals = train_data


        inputs = torch.Tensor(inputs[:, :, :, :in_dim])
        reals = torch.Tensor(reals)

        reals = reals.to(device)
        inputs = inputs.transpose(1, 3)
        inputs = nn.functional.pad(inputs, (1, 0, 0, 0))
        inputs = inputs.to(device)
        optimizer_common.zero_grad()
        optimizer_personal.zero_grad()
        outputs = common_model(inputs)[1]
        predict_common = scaler.inverse_transform(outputs)
        predict_personal = personal_extractor(inputs)[0]

        predict_coupled = adaptive_common_layer(predict_common) + adaptive_personal_layer(predict_personal)


        loss = criterion(predict_coupled, reals)
        loss.backward()
        optimizer_common.step()
        optimizer_personal.step()
        running_loss += loss.item()
    print('Epoch %d loss: %.3f' % (epoch + 1, running_loss / len(train_loader)))

def coupled_validate(scaler, val_loader, common_model, personal_extractor,
                     adaptive_common_layer,adaptive_personal_layer, device, in_dim):
    running_mae = 0.0
    running_rmse = 0.0
    running_mape = 0.0

    running_loss = 0.0
    for val_data in val_loader:
        inputs, reals = val_data


        inputs = torch.Tensor(inputs[:, :, :, :in_dim])
        reals = torch.Tensor(reals)

        reals = reals.to(device)
        inputs = inputs.transpose(1, 3)
        inputs = nn.functional.pad(inputs, (1, 0, 0, 0))
        inputs = inputs.to(device)

        outputs = common_model(inputs)[1]
        predict_common = scaler.inverse_transform(outputs)
        predict_personal = personal_extractor(inputs)[0]

        predict_coupled = adaptive_common_layer(predict_common)+adaptive_personal_layer(predict_personal)
        mae, rmse, mape = get_mae_rmse_mape(predict_coupled, reals)
        running_mae += mae
        running_mape += mape
        running_rmse += rmse
    print('val mae:%.3f, val rmse:%.3f, val mape:%.3f' % (running_mae / len(val_loader),running_rmse/len(val_loader),running_mape/len(val_loader)))


def coupled_test(scaler, test_loader,common_model, personal_extractor,
                 adaptive_common_layer, adaptive_personal_layer,device, in_dim):
    running_mae = 0.0
    running_rmse = 0.0
    running_mape = 0.0

    running_loss = 0.0
    for test_data in test_loader:
        inputs, reals = test_data


        inputs = torch.Tensor(inputs[:, :, :, :in_dim])
        reals = torch.Tensor(reals)

        reals = reals.to(device)
        inputs = inputs.transpose(1, 3)
        inputs = nn.functional.pad(inputs, (1, 0, 0, 0))
        inputs = inputs.to(device)

        outputs = common_model(inputs)[1]
        predict_common = scaler.inverse_transform(outputs)
        predict_personal = personal_extractor(inputs)[0]

        predict_coupled = adaptive_common_layer(predict_common)+adaptive_personal_layer(predict_personal)
        mae, rmse, mape = get_mae_rmse_mape(predict_coupled, reals)
        running_mae += mae
        running_mape += mape
        running_rmse += rmse
    print('test mae:%.3f, test rmse:%.3f, test mape:%.3f' % (running_mae / len(test_loader), running_rmse / len(test_loader), running_mape / len(test_loader)))