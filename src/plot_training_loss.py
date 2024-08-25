import matplotlib.pyplot as plt
import pickle
# upload the training loss data

path = "trained_model_parameters/training_loss_profile_more_layers20240825-003011.pkl"
with open(path , 'rb') as f:
    training_loss = pickle.load(f)


plt.plot(training_loss, marker='o')
ax = plt.gca()
ax.set_xticks(range(1, len(training_loss)+1, 2))
ax.set_xlabel('Training Epoch')
ax.set_ylabel('Loss')
ax.grid()
plt.show()