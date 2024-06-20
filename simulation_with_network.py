import norse
import torch
import matplotlib.pyplot as plt
import support_module as sm




# Define the Network class
class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        
        
        time_constant1 = torch.nn.Parameter(torch.tensor([200.]))
        time_constant2 = torch.nn.Parameter(torch.tensor([100.]))
        time_constant3 = torch.nn.Parameter(torch.tensor([50.]))
        
        voltage1 = torch.nn.Parameter(torch.tensor([0.006]))
        voltage2 = torch.nn.Parameter(torch.tensor([0.08]))
        voltage3 = torch.nn.Parameter(torch.tensor([0.13]))


        # Define three different neuron layers with varying temporal dynamics
        lif_params_1 = norse.torch.LIFBoxParameters(tau_mem_inv= time_constant1 ,v_th = voltage1 )
        lif_params_2 = norse.torch.LIFBoxParameters(tau_mem_inv= time_constant2 ,v_th = voltage2 )
        lif_params_3 = norse.torch.LIFBoxParameters(tau_mem_inv= time_constant3 ,v_th = voltage3 )

        self.temporal_layer_1 = norse.torch.Lift(norse.torch.LIFBoxCell(p=lif_params_1))
        self.temporal_layer_2 = norse.torch.Lift(norse.torch.LIFBoxCell(p=lif_params_2))
        self.temporal_layer_3 = norse.torch.Lift(norse.torch.LIFBoxCell(p=lif_params_3))
        
        self.temporal_layer_1.register_parameter("time_constant",time_constant1)
        self.temporal_layer_1.register_parameter("voltage",voltage1)
        
        self.temporal_layer_2.register_parameter("time_constant",time_constant2)
        self.temporal_layer_2.register_parameter("voltage",voltage2)
        
        self.temporal_layer_3.register_parameter("time_constant",time_constant3)
        self.temporal_layer_3.register_parameter("voltage",voltage3)
    
        
        
        # First convolutional layer
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        
        # Third convolutional layer
        self.linear = torch.nn.Linear(in_features=26,out_features=2)
        
    def forward(self, inputs):
        
        outputs = []
        if len(inputs.shape) == 2: # to deal with a batch
            
            inputs = inputs.unsqueeze(0)
        if len(inputs.shape) == 1: 
            inputs = inputs.unsqueeze(0)
            inputs = inputs.unsqueeze(2)
        
        for input in inputs:
            # made to accept a batch of input
            response_1, response_2, response_3 = [], [], []


            response_1, state_1 = self.temporal_layer_1(input)
            response_2, state_2 = self.temporal_layer_2(input)
            response_3, state_3 = self.temporal_layer_3(input)
            output = torch.stack([response_1, response_2, response_3], dim=0)
            output = self.conv1(output)

            output = torch.transpose(output, 1, 2)

            output = self.linear(output)
            output = torch.transpose(output, 1, 2)


            outputs += [output.squeeze(0)]
        
        if  inputs.shape[0] == 1:
            return outputs[0]
        else :
            return torch.stack(outputs, dim=0) # return the batch
        

# Create an instance of the Network class
network = Network()
# load the model from file
network.load_state_dict(torch.load('model.pth'))

# create multiple simulations
number_of_simulations  = 100
time_samples = 3000
radius       = 6
n_agents     = 7


sm.simulate_neuromorphic_controller(controller=network,
                                    time_samples =time_samples,
                                    radius       = radius,
                                    n_agents     = n_agents,
                                    which_agents =[1],
                                    show_figure  =True)

plt.show()