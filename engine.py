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
    #torch.autograd.set_detect_anomaly(True)
    running_loss = 0.0
    count = 0
    for (base_data, contrastive_data) in zip(base_train_loader,contrastive_train_loader):
        base_inputs, base_reals = base_data
        #print(base_inputs.shape)
        contrastive_inputs, contrastive_reals = contrastive_data
        if len(base_inputs) == len(contrastive_inputs) and len(base_inputs) == batch_size:
            #print(contrastive_inputs.shape)
            base_inputs = torch.Tensor(base_inputs[:, :, :, :in_dim])
            contrastive_inputs = torch.Tensor(contrastive_inputs[:, :, :, :in_dim])
            base_inputs = base_inputs.transpose(1, 3)
            base_inputs = nn.functional.pad(base_inputs, (1, 0, 0, 0))
            base_inputs = base_inputs.to(device)  # input shape: torch.Size([32, 3, 220, 13])
            contrastive_inputs = contrastive_inputs.transpose(1, 3)
            contrastive_inputs = nn.functional.pad(contrastive_inputs, (1, 0, 0, 0))
            contrastive_inputs = contrastive_inputs.to(device)  # input shape: torch.Size([32, 3, 220, 13])
            optimizer.zero_grad()
            base_outputs = personal_extractor(base_inputs)[0]
            #base_outputs = base_scaler.inverse_transform(base_outputs)
            #print(base_outputs.data == last_base_outputs.data)
            if count>0:
                #print(base_outputs.data == last_base_outputs.data)
                #print(base_outputs.data == last_base_outputs.data)
                optimizer.zero_grad()
                loss = max(torch.tensor(2,dtype=torch.float,requires_grad=True),criterion(last_base_outputs,base_outputs))
                #print(loss)
                running_loss+=loss.item()
                if loss>1:
                    loss.backward()
                    optimizer.step()
            contrastive_outputs = personal_extractor(contrastive_inputs)[0]
            #contrastive_outputs = contrastive_scaler.inverse_transform(contrastive_outputs)
            if count>0:
                optimizer.zero_grad()
                loss = max(torch.tensor(2,dtype=torch.float,requires_grad=True),criterion(last_contrastive_outputs,contrastive_outputs))
                #print(loss)
                running_loss+=loss.item()
                if loss>1:
                    loss.backward()
                    optimizer.step()
            base_outputs = personal_extractor(base_inputs)[0]
            #base_outputs = base_scaler.inverse_transform(base_outputs)
            contrastive_outputs = personal_extractor(contrastive_inputs)[0]
            #contrastive_outputs = contrastive_scaler.inverse_transform(contrastive_outputs)
            optimizer.zero_grad()
            loss = 1e4/criterion(base_outputs,contrastive_outputs)
            running_loss-=criterion(base_outputs,contrastive_outputs).item()
            loss.backward()
            optimizer.step()
            #print(base_outputs.shape)
            # base_outputs = base_scaler.inverse_transform(base_outputs)
            # contrastive_outputs = contrastive_scaler.inverse_transform(contrastive_outputs)
            # if count > 0:
                # loss = (criterion(last_base_outputs,base_outputs)
                #         +lambda1*criterion(last_contrastive_outputs,contrastive_outputs)
                #         -lambda2*10*criterion(base_outputs,contrastive_outputs))
                # loss = criterion(last_base_outputs,base_outputs)
                # loss.backward(retain_graph=True)
                # #optimizer.step()
                # print('ok1')
                # #optimizer.zero_grad()
                # # running_loss += loss.item()
                # loss = criterion(last_contrastive_outputs,contrastive_outputs)
                # loss.backward(retain_graph=True)
                # #optimizer.step()
                # print('ok2')
                # #optimizer.zero_grad()
                # # running_loss += loss.item()
                # loss = 100-criterion(base_outputs,contrastive_outputs)
                # loss.backward()
                # optimizer.step()
                # optimizer.step()
                # running_loss += loss.item()
            # if count>0:
            #     #print(last_base_outputs.data == base_outputs.data)
            #     print('similar:',criterion(last_base_outputs,base_outputs),criterion(last_contrastive_outputs,contrastive_outputs))
            #     print('dissimilar:',criterion(base_outputs,contrastive_outputs))
                #loss.backward(retain_graph=True)
                #optimizer.step()
                #running_loss += loss.item()
            count = 1
            last_base_outputs = copy.deepcopy(base_outputs.data)
            last_contrastive_outputs = copy.deepcopy(contrastive_outputs.data)
            # base_outputs+=1
            # contrastive_outputs+=1
            #print(last_base_outputs.data == base_outputs.data)

    print('Epoch %d loss: %.3f' % (epoch + 1, running_loss / len(base_train_loader)))


def train(scaler,train_loader,epoch,optimizer,criterion,model,device,in_dim,last_param=1,last_out=torch.zeros([1,1]),mode='mape'):
    last_out = last_out.to(device)
    #last_param.to(device)
    if mode == 'mape':
        last_param = 1-last_param
    last_param *= 0.01
    total_activate_num = 0
    total_forward_num = 0
    #model.train()
    running_loss = 0.0
    for train_data in train_loader:
        total_forward_num += 1
        model.train()
        inputs, reals = train_data
        # print(inputs[...,:3].shape)
        # inputs = torch.Tensor(np.expand_dims(inputs[:,:,:,:3], axis=-1))  # (32,12,220,1)
        inputs = torch.Tensor(inputs[:, :, :, :in_dim])  # (32,12,220,3)
        # num_nodes = inputs.shape[2]
        # in_dim = inputs.shape[-1]
        # reals = torch.Tensor(np.expand_dims(reals[:,:,:,:3],axis=-1))  # 同上
        # print(reals.shape)
        reals = torch.Tensor(reals)  # (32,12,220,1)
        # print(reals.shape)
        reals = reals.to(device)
        inputs = inputs.transpose(1, 3)
        inputs = nn.functional.pad(inputs, (1, 0, 0, 0))
        inputs = inputs.to(device)  # input shape: torch.Size([32, 3, 220, 13])
        optimizer.zero_grad()
        _, outputs, activate_num = model(inputs) # output shape: torch.Size([32, 12, 220, 1])
        total_activate_num += activate_num
        predict = scaler.inverse_transform(outputs)
        # print(outputs.shape)
        if last_out.shape!=reals.shape:
            last_out = reals
        # scaler...
        target = last_param * last_out + (1 - last_param) * reals
        #loss = criterion(predict, reals)
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
            # print(inputs[...,:3].shape)
            # inputs = torch.Tensor(np.expand_dims(inputs[:,:,:,:3], axis=-1))  # (32,12,220,1)
            inputs = torch.Tensor(inputs[:, :, :, :in_dim])  # (32,12,220,3)
            # num_nodes = inputs.shape[2]
            # in_dim = inputs.shape[-1]
            # reals = torch.Tensor(np.expand_dims(reals[:,:,:,:3],axis=-1))  # 同上
            # print(reals.shape)
            reals = torch.Tensor(reals)  # (32,12,220,1)
            # print(reals.shape)
            reals = reals.to(device)
            inputs = inputs.transpose(1, 3)
            inputs = nn.functional.pad(inputs, (1, 0, 0, 0))
            inputs = inputs.to(device)  # input shape: torch.Size([32, 3, 220, 13])
            _,outputs,_ = model(inputs) # output shape: torch.Size([32, 12, 220, 1])
            predict = scaler.inverse_transform(outputs)
            #loss = criterion(predict, reals)
            mae,rmse,mape = get_mae_rmse_mape(predict, reals)
            #running_loss += loss.item()
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
            # print(inputs[...,:3].shape)
            # inputs = torch.Tensor(np.expand_dims(inputs[:,:,:,:3], axis=-1))  # (32,12,220,1)
            inputs = torch.Tensor(inputs[:, :, :, :in_dim])  # (32,12,220,3)
            # num_nodes = inputs.shape[2]
            # in_dim = inputs.shape[-1]
            # reals = torch.Tensor(np.expand_dims(reals[:,:,:,:3],axis=-1))  # 同上
            # print(reals.shape)
            reals = torch.Tensor(reals)  # (32,12,220,1)
            # print(reals.shape)
            reals = reals.to(device)
            inputs = inputs.transpose(1, 3)
            inputs = nn.functional.pad(inputs, (1, 0, 0, 0))
            inputs = inputs.to(device)  # input shape: torch.Size([32, 3, 220, 13])
            outputs = model(inputs)[1]  # output shape: torch.Size([32, 12, 220, 1])
            predict = scaler.inverse_transform(outputs)
            # loss = criterion(predict, reals)
            mae, rmse, mape = get_mae_rmse_mape(predict, reals)
            # running_loss += loss.item()
            running_mae += mae
            running_mape += mape
            running_rmse += rmse
    print('test mae:%.3f, test rmse:%.3f, test mape:%.3f' % (
        running_mae / len(test_loader), running_rmse / len(test_loader), running_mape / len(test_loader)))
    return running_mae / len(test_loader), running_rmse / len(test_loader), running_mape / len(test_loader)


def coupled_train(scaler, train_loader, epoch, optimizer_common, optimizer_personal, criterion, common_model, personal_extractor, adaptive_common_layer, adaptive_personal_layer, device, in_dim):
    #model.train()
    running_loss = 0.0
    for train_data in train_loader:
        #model.train()
        # common_model.no_grad()
        # personal_extractor.no_grad()
        adaptive_common_layer.train()
        inputs, reals = train_data
        # print(inputs[...,:3].shape)
        # inputs = torch.Tensor(np.expand_dims(inputs[:,:,:,:3], axis=-1))  # (32,12,220,1)
        inputs = torch.Tensor(inputs[:, :, :, :in_dim])  # (32,12,220,3)
        # num_nodes = inputs.shape[2]
        # in_dim = inputs.shape[-1]
        # reals = torch.Tensor(np.expand_dims(reals[:,:,:,:3],axis=-1))  # 同上
        # print(reals.shape)
        reals = torch.Tensor(reals)  # (32,12,220,1)
        # print(reals.shape)
        reals = reals.to(device)
        inputs = inputs.transpose(1, 3)
        inputs = nn.functional.pad(inputs, (1, 0, 0, 0))
        inputs = inputs.to(device)  # input shape: torch.Size([32, 3, 220, 13])
        optimizer_common.zero_grad()
        optimizer_personal.zero_grad()
        outputs = common_model(inputs)[1]  # output shape: torch.Size([32, 12, 220, 1])
        predict_common = scaler.inverse_transform(outputs)
        predict_personal = personal_extractor(inputs)[0]
        #concatenated = torch.cat([predict_common,predict_personal],dim=1)
        predict_coupled = adaptive_common_layer(predict_common) + adaptive_personal_layer(predict_personal)
        # print(outputs.shape)
        # scaler...
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
    # model.train()
    running_loss = 0.0
    for val_data in val_loader:
        # model.train()
        # common_model.no_grad()
        # personal_extractor.no_grad()
        #adaptive_layer.train()
        inputs, reals = val_data
        # print(inputs[...,:3].shape)
        # inputs = torch.Tensor(np.expand_dims(inputs[:,:,:,:3], axis=-1))  # (32,12,220,1)
        inputs = torch.Tensor(inputs[:, :, :, :in_dim])  # (32,12,220,3)
        # num_nodes = inputs.shape[2]
        # in_dim = inputs.shape[-1]
        # reals = torch.Tensor(np.expand_dims(reals[:,:,:,:3],axis=-1))  # 同上
        # print(reals.shape)
        reals = torch.Tensor(reals)  # (32,12,220,1)
        # print(reals.shape)
        reals = reals.to(device)
        inputs = inputs.transpose(1, 3)
        inputs = nn.functional.pad(inputs, (1, 0, 0, 0))
        inputs = inputs.to(device)  # input shape: torch.Size([32, 3, 220, 13])
        #optimizer.zero_grad()
        outputs = common_model(inputs)[1]  # output shape: torch.Size([32, 12, 220, 1])
        predict_common = scaler.inverse_transform(outputs)
        predict_personal = personal_extractor(inputs)[0]
        #concatenated = torch.cat([predict_common,predict_personal],dim=1)
        predict_coupled = adaptive_common_layer(predict_common)+adaptive_personal_layer(predict_personal)
        # print(outputs.shape)
        # scaler...
        # loss = criterion(predict_coupled, reals)
        # loss.backward()
        # optimizer.step()
        #running_loss += loss.item()
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
    # model.train()
    running_loss = 0.0
    for test_data in test_loader:
        # model.train()
        # common_model.no_grad()
        # personal_extractor.no_grad()
        #adaptive_layer.train()
        inputs, reals = test_data
        # print(inputs[...,:3].shape)
        # inputs = torch.Tensor(np.expand_dims(inputs[:,:,:,:3], axis=-1))  # (32,12,220,1)
        inputs = torch.Tensor(inputs[:, :, :, :in_dim])  # (32,12,220,3)
        # num_nodes = inputs.shape[2]
        # in_dim = inputs.shape[-1]
        # reals = torch.Tensor(np.expand_dims(reals[:,:,:,:3],axis=-1))  # 同上
        # print(reals.shape)
        reals = torch.Tensor(reals)  # (32,12,220,1)
        # print(reals.shape)
        reals = reals.to(device)
        inputs = inputs.transpose(1, 3)
        inputs = nn.functional.pad(inputs, (1, 0, 0, 0))
        inputs = inputs.to(device)  # input shape: torch.Size([32, 3, 220, 13])
        #optimizer.zero_grad()
        outputs = common_model(inputs)[1]  # output shape: torch.Size([32, 12, 220, 1])
        predict_common = scaler.inverse_transform(outputs)
        predict_personal = personal_extractor(inputs)[0]
        #concatenated = torch.cat([predict_common,predict_personal],dim=1)
        predict_coupled = adaptive_common_layer(predict_common)+adaptive_personal_layer(predict_personal)
        # print(outputs.shape)
        # scaler...
        # loss = criterion(predict_coupled, reals)
        # loss.backward()
        # optimizer.step()
        #running_loss += loss.item()
        mae, rmse, mape = get_mae_rmse_mape(predict_coupled, reals)
        running_mae += mae
        running_mape += mape
        running_rmse += rmse
    print('test mae:%.3f, test rmse:%.3f, test mape:%.3f' % (running_mae / len(test_loader), running_rmse / len(test_loader), running_mape / len(test_loader)))