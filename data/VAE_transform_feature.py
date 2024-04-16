import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import random_split
from copy import deepcopy
from model import VAE

import matplotlib.pyplot as plt
from load_data import load_data_1D_impute   

class MyDataset(Dataset):
    
    def __init__(self,X,y):
        self.X=X
        self.y=y

    def __getitem__(self,index):
        sample={'data':self.X[index],'label':self.y[index]}
        return(sample)
    
    def __len__(self):
        return len(self.y)
    
    
def VAE_transform(data_dir="/mnt/binf/eric/Mercury_Dec2023/Feature_all_Mar2024.pkl", input_size=950, feature_type="Arm", encoding_size=64, control="Yes", alpha=0.5, beta=0.5, num_epochs = 200, num_repeats = 10):

    ### alpha controls the weight of latent space loss; beta controls the weight of task loss
    torch.autograd.set_detect_anomaly(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ### load data; use X_r01b as AE fitting data
    data, X_train_tensor, y_train_tensor, _, _, _, _, X_all_tensor, _, _, X_r01b_tensor, y_r01b_tensor, _ = load_data_1D_impute(data_dir, input_size, feature_type) 
    
    X_train_tensor = torch.squeeze(X_train_tensor)
    X_all_tensor = torch.squeeze(X_all_tensor)
    X_r01b_tensor = torch.squeeze(X_r01b_tensor)
    
    r01b_dataset = MyDataset(X_r01b_tensor,y_r01b_tensor)
    
    train_num = int(X_r01b_tensor.shape[0] * 0.75)
    test_num = X_r01b_tensor.shape[0] - train_num
    
    X_r01b_train, X_r01b_test = torch.split(X_r01b_tensor,[train_num, test_num])
    y_r01b_train = torch.ones(X_r01b_train.shape[0])
    y_r01b_test = torch.ones(X_r01b_test.shape[0])
    # r01b_dataset_train, r01b_dataset_test = random_split(r01b_dataset,[train_num, test_num])
    # X_r01b_train = torch.stack([sample['data'] for sample in r01b_dataset_train])
    # y_r01b_train = torch.stack([sample['label'] for sample in r01b_dataset_train])

    # X_r01b_test = torch.stack([sample['data'] for sample in r01b_dataset_test])
    # y_r01b_test = torch.stack([sample['label'] for sample in r01b_dataset_test])

    
    if "Yes" in control:
        
        print("******************  Add 70 KAG9 healthy samples into VAE fitting  ******************")
        X_train_healthy = X_train_tensor[y_train_tensor == 0]
        X_train_healthy70 = X_train_healthy[:70]
        
        y_r01b_train = torch.cat((torch.ones(X_r01b_train.shape[0]), torch.zeros(52)),dim=0)
        y_r01b_test = torch.cat((torch.ones(X_r01b_test.shape[0]), torch.zeros(18)),dim=0)
        
        X_healthy70_train, X_healthy70_test = torch.split(X_train_healthy70,[52, 18])
        
        X_r01b_train = torch.cat((X_r01b_train,X_healthy70_train),dim=0)
        X_r01b_test = torch.cat((X_r01b_test,X_healthy70_test),dim=0)
    
    else:
        print("******************  Only R01B-match samples are used in VAE fitting  ******************")    

    X_all_tensor = X_all_tensor.to(device)        
    X_r01b_train=X_r01b_train.to(device)
    X_r01b_test=X_r01b_test.to(device)
    y_r01b_train=y_r01b_train.to(device)
    y_r01b_test=y_r01b_test.to(device)
        
    # Instantiate the autoencoder
    input_size = X_r01b_tensor.shape[1]
    encoding_size = 64
    
    Variational_AE = VAE(input_size, encoding_size)
    Variational_AE.to(device)
    
    # Loss function and optimizer
    criterion_recons = nn.MSELoss()
    criterion_latent = nn.KLDivLoss(reduction="batchmean")
    criterion_task = nn.BCELoss(reduction="mean")
    
    parameters_VAE = [] # exclude task_classifier, domain_classifier, and task_classifier2
    for name, param in Variational_AE.named_parameters():
        if not 'classifier' in name:
            print(name)
            parameters_VAE.append(param)
    
    # optimizer_VAE = optim.Adam([{'params':parameters_VAE}], lr=1e-5, weight_decay=1e-6)
    optimizer_VAE = optim.Adam(Variational_AE.parameters(), lr=1e-5, weight_decay=1e-6)
    optimizer_taskclassifier = optim.SGD(Variational_AE.task_classifier.parameters(), lr=1e-5, weight_decay=1e-6)
    optimizer_taskclassifier2 = optim.Adam(Variational_AE.task_classifier2.parameters(), lr=1e-4, weight_decay=1e-6)

    # store loss values through epoches
    train_loss_list = []
    test_loss_list = []
    
    train_recons_loss_list = []
    train_latent_loss_list = []
    train_task_loss_list = []
    train_task2_loss_list = []
    
    test_recons_loss_list = []
    test_latent_loss_list = []
    test_task_loss_list = []    
    test_task2_loss_list = [] 
    
    # Training the autoencoder
    num_epochs = num_epochs
    min_loss = float("inf")
    patience = 50
    
    for epoch in range(num_epochs):
        
        seed = 42 + epoch
        shuffled_indices = torch.randperm(X_r01b_train.size(0))
        X_r01b_train = X_r01b_train[shuffled_indices]
        y_r01b_train = y_r01b_train[shuffled_indices]
        
        decodings,code_mean,code_sd,_,code_task,_,output_task = Variational_AE(X_r01b_train,alpha=0.01)
            
        # print("==========================================")
        # print(torch.sum(torch.isnan(code_task)).item())
               
        loss_recons = criterion_recons(decodings,X_r01b_train)
        # loss_latent = criterion_latent(F.log_softmax(code_sd, dim=1),F.softmax(code_mean, dim=1))
        loss_latent = torch.distributions.kl_divergence(torch.distributions.Normal(loc=code_mean,scale=torch.exp(code_sd)),
                                       torch.distributions.Normal(0,1.0)).sum(-1).mean()

        loss_task = criterion_task(code_task,y_r01b_train)
        loss_task2 = criterion_task(output_task,y_r01b_train)
        
        loss = loss_recons + alpha*loss_latent + beta*loss_task + beta*loss_task2
        # loss = loss_recons + 0.25*loss_task
        
        optimizer_VAE.zero_grad()
        optimizer_taskclassifier.zero_grad()
        optimizer_taskclassifier2.zero_grad()
    
        loss.backward(retain_graph = True)
        
        optimizer_taskclassifier.step()
        optimizer_taskclassifier2.step()
        optimizer_VAE.step()
        
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        
        # Evaluation on test data
        with torch.no_grad():
            Variational_AE.eval()
            
            test_decodings,test_mean,test_sd,_,test_task,_,test_output_task=Variational_AE.forward(X_r01b_test,alpha=0.01)
            
           
            test_loss_recons=criterion_recons(test_decodings, X_r01b_test)
            # test_loss_latent=criterion_latent(F.log_softmax(test_sd, dim=1),F.softmax(test_mean, dim=1))
            test_loss_latent=torch.distributions.kl_divergence(torch.distributions.Normal(loc=test_mean,scale=torch.exp(test_sd)),
                                       torch.distributions.Normal(0,1)).sum(-1).mean()

            
            test_loss_task=criterion_task(test_task,y_r01b_test)
            test_loss_task2=criterion_task(test_output_task,y_r01b_test)
            # test_loss = test_loss_recons + alpha*test_loss_latent + beta*test_loss_task + beta*test_loss_task2
            test_loss = test_loss_recons + alpha*test_loss_latent
            
            print(f"========= Epoch {epoch} =========")
            print(f"train loss: {loss.item():.4f}, test loss: {test_loss.item():.4f}")
            print(f"test reconstruction loss: {test_loss_recons.item():.4f}")
            print(f"test latent space loss: {test_loss_latent.item():.4f}")
            print(f"test task loss: {test_loss_task.item():.4f}")
            print(f"test task2 loss: {test_loss_task2.item():.4f}")
            
            train_loss_list.append(loss.item())
            test_loss_list.append(test_loss.item())
            
            train_recons_loss_list.append(loss_recons.item())
            train_latent_loss_list.append(loss_latent.item())
            train_task_loss_list.append(loss_task.item())
            train_task2_loss_list.append(loss_task2.item())
            
            test_recons_loss_list.append(test_loss_recons.item())
            test_latent_loss_list.append(test_loss_latent.item())
            test_task_loss_list.append(test_loss_task.item())
            test_task2_loss_list.append(test_loss_task2.item())
            
            # Early stopping check
            if test_loss <= min_loss:
                min_loss = test_loss
                best_model = deepcopy(Variational_AE.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping triggered! No improvement in {patience} epochs.")
                    break
    
    # Plotting the loss values
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_loss_list) + 1), train_loss_list, label='Train Loss')
    plt.plot(range(1, len(test_loss_list) + 1), test_loss_list, label='Test Loss')
    
    plt.plot(range(1, len(train_recons_loss_list) + 1), [x * 5 for x in train_recons_loss_list], label='Train Reconstruction Loss')
    plt.plot(range(1, len(train_latent_loss_list) + 1), [x * 10 for x in train_latent_loss_list], label='Train Latent Space Loss')
    plt.plot(range(1, len(train_task_loss_list) + 1), train_task_loss_list, label='Train Task Loss')
    plt.plot(range(1, len(train_task2_loss_list) + 1), train_task2_loss_list, label='Train Task2 Loss')
   
    plt.plot(range(1, len(test_recons_loss_list) + 1), [x * 5 for x in test_recons_loss_list], label='Test Reconstruction Loss')
    plt.plot(range(1, len(test_latent_loss_list) + 1), [x * 10 for x in test_latent_loss_list], label='Test Latent Space Loss')
    plt.plot(range(1, len(test_task_loss_list) + 1), test_task_loss_list, label='Test Task Loss')
    plt.plot(range(1, len(test_task2_loss_list) + 1), test_task2_loss_list, label='Test Task2 Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Losses over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()
        
    # Encode data using the trained autoencoder
    Variational_AE.train()
    Variational_AE.min_loss = min_loss
    Variational_AE.load_state_dict(best_model)
    
    variable_name_list = [f"VAE_{feature_type}_{i}" for i in range(encoding_size)]
    
    print("----------------------------------------------------")
    print(len(variable_name_list))
    
    data_return = None
    for i in range(num_repeats):
        
        print(f"---------------- {i} ----------------")
        _,_,_,code_all,_,_,_ = Variational_AE(X_all_tensor, alpha=0.01)
        encoded_all = pd.DataFrame(code_all.detach().cpu().numpy(), columns=variable_name_list)
        
        data_VAE = pd.concat([data.loc[:,['SampleID','Train_Group','train','Project','R01B_label']],encoded_all],axis=1)

        if i == 0:
            data_return = data_VAE.copy()
        else:
            data_VAE_R01B = data_VAE.loc[data_VAE["R01B_label"] == "R01B_match"]
            data_VAE_R01B.loc[:,"SampleID"] =  data_VAE_R01B.loc[:,"SampleID"] + f"-simu{i}"

            data_return = pd.concat([data_return, data_VAE_R01B],axis=0,ignore_index=True)
            
    print(f"The feature data to return contains {data_return.loc[data_return['Project'] == 'R01B'].shape[0]} R01B samples")
                   
    return data_return




