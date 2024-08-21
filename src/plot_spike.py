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

print("features shape")
print(features.shape, labels.shape)

n_time_points = features[0].shape[1]
# Apply the standard scaling to the dataset
scaler   = sm.StandardScaler()
features = scaler.fit_transform(features[:-1:30,:,:])
labels   = labels[:-1:30,:,:]


network = sm.Network(train_mode=True)



outputs = network(features[10,:,:])
network.plot_latest_responses()


