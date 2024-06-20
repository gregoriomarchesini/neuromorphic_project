# neuromorphic_project

`support_module.py`   : contains the main routines to run the simulations and store the data
`create_dataset.py`   : starts simulations and extracts the data to create the dataset for the training. A dataset folder si created from this script
`network_training.py` : after running `create_dataset.py` you can train the network with this function.
`simulation_with_network` : simulate some agents controller with a standard controller and other simulated with the neuromorphic controller
