import support_module as sm
from logging import basicConfig, INFO,ERROR
import pickle as pkl
import matplotlib.pyplot as plt
import os,sys
from tqdm import tqdm

basicConfig(filename='simulator.log', level=ERROR, filemode='w')

###############################
############ Choose here
#################################
save_dataset = False

# create multiple simulations
number_of_simulations  = 1
time_samples = 10000
radius       = 7
n_agents     = 7
counter      = 0


complete_dataset = {}
for i in tqdm(range(number_of_simulations)) :
    dataset_per_agent = sm.create_dataset(radius       = radius, 
                                          n_agents     = n_agents,
                                          time_samples = time_samples, 
                                          show_figure  = True)
    
    # saving the numpy arrays in folder 
    for data in  dataset_per_agent.values():
        complete_dataset[counter] = data
        counter += 1


if save_dataset :
    # directory of the dataset 
    dataset_dir = "dataset"
    os.makedirs(dataset_dir,exist_ok=True)


    with open(dataset_dir + "/dataset.pkl", 'wb') as f:
        pkl.dump(complete_dataset, f)




plt.show()






