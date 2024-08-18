import pickle 
import norse
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
import support_module as sm

with open('dataset/dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)

features = torch.stack([torch.from_numpy(a["feature"]).float() for a in dataset.values()], dim=0)
labels   = torch.stack([torch.from_numpy(a["label"]).float() for a in dataset.values()], dim=0)

print(features.shape, labels.shape)

n_time_points = features[0].shape[1]
# Apply the standard scaling to the dataset
scaler   = sm.StandardScaler()
features = scaler.fit_transform(features[:-1:30,:,:])
labels   = labels[:-1:30,:,:]


class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return feature, label

# Create the dataset
dataset = CustomDataset(features, labels)

# Split the dataset into train and test sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=15, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=30, shuffle=False)

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
patience   = 5
best_loss = float('inf')
epochs_no_improve = 0

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
    
    # Early stopping
    if train_loss < best_loss:
        best_loss = train_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
    
    if epochs_no_improve >= patience:
        print(f'Early stopping at epoch {epoch+1}')
        break

# Evaluation on the test set
network.eval()
with torch.no_grad():
    test_loss = 0
    for batch_features, batch_labels in test_loader:
        outputs = network(batch_features)
        loss = criterion(outputs, batch_labels)
        test_loss += loss.item()
    
    test_loss /= len(test_loader)
    print(f'Test Loss: {test_loss}')

torch.save(network.state_dict(), 'model.pth')
scaler.save('scaler_params.pkl')
