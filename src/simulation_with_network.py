import torch
import matplotlib.pyplot as plt
import support_module as sm


# Create an instance of the Network class
network = sm.Network(train_mode=False)
# load the model from file
network.load_state_dict(torch.load('trained_model_parameters/model_new20240820-123235.pth'))
scaler = sm.StandardScaler()
scaler.load('trained_model_parameters/scaler_params_new20240820-123235.pkl')


# create multiple simulations
number_of_simulations  = 100
time_samples = 10000
radius       = 7
n_agents     = 7


sm.simulate_neuromorphic_controller(neuromorphic_controller=network,
                                    time_samples =time_samples,
                                    scaler=scaler,
                                    radius       = radius,
                                    n_agents     = n_agents,
                                    neuromorphic_agents =[5],
                                    show_figure  =True)

plt.show()