import pickle 
import norse
import torch
import matplotlib.pyplot as plt
from torch.utils.data import  DataLoader, random_split
import support_module as sm
import random
import time 
import os


save_training = True
with open('dataset/dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)


features = torch.stack([torch.from_numpy(a["feature"]).float() for a in dataset.values()], dim=0)
labels   = torch.stack([torch.from_numpy(a["label"]).float() for a in dataset.values()], dim=0)

print(features.shape, labels.shape)
features = features[random.sample(range(features.shape[0]), 300),:,::3]
labels   = labels[random.sample(range(labels.shape[0]), 300),:,::3]

n_time_points = features[0].shape[1]

# Create the dataset
dataset = sm.CustomDataset(features, labels)

# Split the dataset into train and test sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

scaler = sm.StandardScaler()

# Fit the scaler on the features of the training dataset and transform them
train_features = scaler.fit_transform(train_dataset.dataset.features[train_dataset.indices])

# Transform the features of the test dataset using the fitted scaler
test_features = scaler.transform(test_dataset.dataset.features[test_dataset.indices])
print(train_features.shape, test_features.shape)

# Update the datasets with the scaled features
train_dataset.dataset.features[train_dataset.indices] = train_features
test_dataset.dataset.features[test_dataset.indices] = test_features

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=30, shuffle=True)

def loss_fn(predicted_optimal_inputs, computed_optimal_inputs):
    # expects a batch
    cost = 0
    batch_len = predicted_optimal_inputs.shape[0]
    for jj in  range(batch_len):
        cost += torch.sum((predicted_optimal_inputs[jj] -  computed_optimal_inputs[jj])**2)
    return cost / batch_len / n_time_points # control input error at each time instant

network = sm.Network(train_mode=True)
criterion = loss_fn
optimizer = torch.optim.Adam(network.parameters(), lr=0.02)

# Training loop with early stopping
num_epochs = 80
patience   = 10
best_loss = float('inf')
epochs_no_improve = 0
loss_history = []
best_model_state_dict = None  # Variable to store the best model parameters

for epoch in range(num_epochs):
    network.train()
    train_loss = 0
    for batch_features, batch_labels in train_loader:
        optimizer.zero_grad()
        outputs = network(batch_features)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss}')
    loss_history += [train_loss]
    
    # Early stopping
    if train_loss < best_loss:
        best_loss = train_loss
        epochs_no_improve = 0
        best_model_state_dict = network.state_dict()  # Save the best model state dict
    else:
        epochs_no_improve += 1
    
    if epochs_no_improve >= patience:
        print(f'Early stopping at epoch {epoch+1}')
        break

# If early stopping was triggered or training completed, save the best model parameters
if best_model_state_dict is not None:
    network.load_state_dict(best_model_state_dict)  # Load the best model state dict

# Save the best model parameters and related data if save_training is True
if save_training:
    parameters_folder = 'trained_model_parameters'
    dt_string = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(parameters_folder, exist_ok=True)
    
    torch.save(best_model_state_dict, parameters_folder + '/model_best_more_layers' + dt_string + '.pth')
    scaler.save(parameters_folder + '/scaler_params_more_layers_' + dt_string + '.pkl')
    with open(parameters_folder + '/training_loss_profile_more_layers' + dt_string + '.pkl', 'wb') as file:
        pickle.dump(loss_history, file)