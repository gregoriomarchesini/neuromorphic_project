
from   collections import namedtuple
from   scipy.optimize import  LinearConstraint, minimize
from   logging import getLogger
import random
from math import cos,sin,pi,atan
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

Identifier = int
Vector     = namedtuple('Vector', ['x', 'y'])
random_between = lambda a,b : a + random.random()*(b-a)

def sum_vectors(vector1:Vector,vector2:Vector):
    return Vector(vector1.x+vector2.x,vector1.y+vector2.y)

def dot(vector1:Vector,vector2:Vector):
    return vector1.x*vector2.x+vector1.y*vector2.y

def norm(vector1:Vector):
    return (vector1.x**2+vector1.y**2)**0.5

def normalize(vector1:Vector):
    norm = (vector1.x**2+vector1.y**2)**0.5
    return Vector(vector1.x/norm,vector1.y/norm)

def scalar_multiplication(vector1:Vector,scalar:float):
    return Vector(vector1.x*scalar,vector1.y*scalar)


def bearing_angle_sin_and_cos(vector1:Vector):
    norm = (vector1.x**2+vector1.y**2)**0.5
    
    if norm == 0:
        return 1,0
    else :
        
        cos_bearing = vector1.x/norm
        sin_bearing = vector1.y/norm
        
        return cos_bearing,sin_bearing



# All agents assumed to have the same max velocity and same type of constraint on the acceleration

class Agent:
    
    time_step           = 0.008
    nominal_velocity    = 3
    max_velocity        = 3 # considered sudden change
    alpha               = 1.4
    collision_radius    = 0.3
    
    def __init__(self,identifier: Identifier, position=Vector(0,0), goal_position = Vector(0,0) ) -> None:
        
        self.identifier = identifier            # identifier for the agent
        self.position   = position              # current position of the agent
        self.velocity   = Vector(0,0)           # current velocity of the agent
        self.other_agents_position_current = {} # each key is an identity for the agent and the value is the position of the agent 
        self.other_agents_position_past    = {} # each key is an identity for the agent and the value is the position of the agent 
        self.other_agents_worse_impact     = {} # each key is an identity for the agent and the value is the worse velocity of the agent
        
        self.neuromorphic_controller = None
        self.neuromorphic_controller_set = False
        
        self.other_agents_velocity = {} # each key is an identity for the agent and the value is the velocity of the agent
        self.position_callbacks    = [] # takes the callbacks from all the other agents
        
        self.goal_position = goal_position
        self.optimal_velocity_change = Vector(0,0)
        
        self.logger = getLogger("Agent_"+str(identifier))
        
    def get_position_callback(self,position:Vector,identity:Identifier):
        """ callback to get the state of the other agents"""
        
        # self.logger.info("Received position from agent "+str(identity))
        self.other_agents_position_past[identity]    = self.other_agents_position_current.setdefault(identity,position) # at the first iteration just set the current and past position as the same
        self.other_agents_position_current[identity] = position
    
    def notify_position(self):
        for callback in self.position_callbacks:
            ## instances of the get_position_callback method from the other agents
            callback(self.position,self.identifier)
    
    def update_other_agents_velocity(self):
        for agent_id in self.other_agents_position_current.keys():
            # self.logger.info("Updating velocity estimate for agent "+str(agent_id))
            past_pos = self.other_agents_position_past[agent_id]
            current_pos = self.other_agents_position_current[agent_id]
            self.other_agents_velocity[agent_id] = Vector((current_pos.x-past_pos.x)/self.time_step,(current_pos.y-past_pos.y)/self.time_step) # estimation of the current velocity of the agent
            
    def set_neuromorphic_controller(self,controller):
        self.neuromorphic_controller = controller
        self.neuromorphic_controller_set = True
        
    def unset_neuromorphic_controller(self):
        self.neuromorphic_controller_set = False
        
    
    def compute_barrier_constraints(self) :
        
        # barrier ->    ||r_me-r_other|| -c_r^2
        
        # dr =  (r_me-r_other)
        # dv =  ( v_me+v_change - (v_other +v_change_other))
        

    
        # so : normal direction = -(r_me-r_other)
        #      b                = -(alpha*(|r_me-r_other|^2-cr^2)  + worse_impact)
        
        normal_vector = []
        b_vector            = []
        
        for agent_id,position_other in self.other_agents_position_current.items():
            
            relative_position = Vector(self.position.x - position_other.x,self.position.y - position_other.y)
            distance_square   = dot(relative_position,relative_position) #|r_me-r_other|^2

            dr = [-(relative_position.x),-(relative_position.y)]
            b  =  -(-self.alpha*(distance_square-self.collision_radius**2)/2 )
            
            normal_vector.append(dr)
            b_vector.append(b)
            
        return normal_vector,b_vector
    
    
    def input_constraint(self):
         # Ax <= b
        A =  [[ 1 ,  0 ],
              [-1 ,  0 ],
              [ 0 , -1 ],
              [ 0 ,  1 ]]
    
        b = [self.max_velocity,self.max_velocity,self.max_velocity,self.max_velocity]
    
        return A,b
    
    
    def current_desired_velocity(self):
        
        direction_to_goal = normalize(Vector(self.goal_position.x-self.position.x,self.goal_position.y-self.position.y))
        distance_to_goal  = norm(Vector(self.goal_position.x-self.position.x,self.goal_position.y-self.position.y))
        
        desired_velocity = scalar_multiplication(direction_to_goal,atan(distance_to_goal)/pi*self.nominal_velocity) 
    
        return desired_velocity
    
    
    
    def cost(self,x, desired_velocity: Vector,current_velocity:Vector):
        """select velocity change that leads the closest to the desired velocity"""
        return    10*(desired_velocity.x - x[0])**2 + 10*(desired_velocity.y - x[1])**2 
    
    
    def cost_jac(self,x, desired_velocity: Vector,current_velocity:Vector) :
            
        return [-2*(desired_velocity.x - (x[0])),
                -2*(desired_velocity.y - (x[1]))]
    
    def compute_optimal_velocity_change(self):
        
        
        if not self.neuromorphic_controller_set:
            A_barrier,b_barrier = self.compute_barrier_constraints()
            A,b = self.input_constraint()
            
            A = A_barrier + A
            b = b_barrier + b
        
            desired_velocity = self.current_desired_velocity()
        
            current_velocity = self.velocity
            x0 = [0.0,0.0]
            
            linear_constraint = LinearConstraint(A, ub =b)
            
            res = minimize(self.cost, x0, 
                        method      = 'SLSQP', 
                        constraints = linear_constraint,
                        jac         = self.cost_jac,
                        args        = (desired_velocity, current_velocity), 
                        options     = {'disp': False,"ftol":1e-3,"maxiter":10})
            

            if res.success:
                self.logger.info("Solution found... speed change : "+str(res.x))
                return Vector(res.x[0],res.x[1])
            else :
                self.logger.error("No solution found")
                return Vector(0,0)
       
        else:
                
            # relative_position_other_agents
            # relative_velocity_other_agents
            # relative_position_from_goal
            
            # the order of the input types matter but not the order of the agents 
            input_data = []
            for _,position_other in self.other_agents_position_current.items():
                input_data += [self.position.x - position_other.x,self.position.y - position_other.y]
            
            for _,velocity_other in self.other_agents_velocity.items():
                input_data += [self.velocity.x - velocity_other.x,self.velocity.y - velocity_other.y]
                
            input_data += [self.goal_position.x-self.position.x,self.goal_position.y-self.position.y]
            
            input = torch.tensor(input_data).float()
            output = self.neuromorphic_controller(input).detach()
                
            self.logger.info("NN output : "+str(output))
            return Vector(float(output[0]),float(output[1]))
 
    def step(self):
        
        
        self.update_other_agents_velocity()
        self.optimal_input = self.compute_optimal_velocity_change()
        self.velocity = self.optimal_input# adding some noise to the velocity
        self.position = sum_vectors(self.position, scalar_multiplication(self.velocity,self.time_step))
        
        


def relative_position_other_agents(agent_id:Identifier,agents:list[Agent],agent_trajectories:dict, agent_velocities:dict):
    """computes the relative position of all the other agents with respect to the agent_id"""
    data = []
    for other_agent_id in agent_trajectories.keys():
        if agent_id != other_agent_id:
            data_agent = []
            for time in agent_trajectories[other_agent_id].keys():
                data_agent.append([agent_trajectories[agent_id][time].x-agent_trajectories[other_agent_id][time].x,
                             agent_trajectories[agent_id][time].y-agent_trajectories[other_agent_id][time].y])
            data.append(data_agent)
    return data


def relative_velocity_other_agents(agent_id:Identifier,agents:list[Agent],agent_trajectories:dict, agent_velocities:dict):
    """Computes the relative velocity of all the other agents with respect to the agent_id"""
    data = []
    for other_agent_id in agent_velocities.keys():
        if agent_id != other_agent_id:
            data_agent = []
            for time in agent_velocities[other_agent_id].keys():
                data_agent.append([agent_velocities[agent_id][time].x-agent_velocities[other_agent_id][time].x,
                             agent_velocities[agent_id][time].y-agent_velocities[other_agent_id][time].y])
            data.append(data_agent)
    return data


def relative_position_from_goal(agent_id:Identifier,agents:list[Agent],agent_trajectories:dict, agent_velocities:dict):
    """Computes the relative position of the agent with respect to its goal position"""
    data = []
    for agent in agents:
        if agent.identifier == agent_id: # only pick the current agent
            for time in agent_trajectories[agent_id].keys():
                data.append([agent.goal_position.x-agent_trajectories[agent_id][time].x,
                             agent.goal_position.y-agent_trajectories[agent_id][time].y])
    return [data]           

def relative_position_from_goal_other_agents(agent_id:Identifier,agents:list[Agent],agent_trajectories:dict, agent_velocities:dict):
    """Computes the relative position of all the other agents with respect to their goal position. (So all relative goal position except the agent_id)"""
    data = []
    for other_agent_id in agent_trajectories.keys():
        if agent_id != other_agent_id:
            data_agent = []
            for time in agent_trajectories[other_agent_id].keys():
                data_agent.append([agents[other_agent_id].goal_position.x-agent_trajectories[other_agent_id][time].x,
                                   agents[other_agent_id].goal_position.y-agent_trajectories[other_agent_id][time].y])
            data.append(data_agent)
    return data
            
def prepare_dataset(agents:list[Agent],agent_trajectories:dict, agent_velocities:dict,agents_input:dict) :

    # combo : relative_position_other_agents,relative_velocity_other_agents,relative_position_from_goal
    
    combo = [relative_position_other_agents,relative_velocity_other_agents,relative_position_from_goal]
    
    data_set_per_agent = {}
    for agentid in agent_trajectories.keys():
        feature = []
        label   = []
        for function in combo:
            feature += function(agentid,agents,agent_trajectories,agent_velocities)
        
        for input in agents_input[agentid].values():
            label.append([input.x,input.y])
        
        feature = np.hstack([np.array(component) for component in feature]).T # time in the x direction and the state on the y
        label   = np.array(label).T
        
        data_set_per_agent[agentid] = {"feature":feature,"label":label}
    
    return data_set_per_agent


def create_dataset(radius, n_agents,time_samples, show_figure:bool = False) :

    start_positions  = [[radius * random_between(0.8,1.2) * cos(2*pi/(n_agents)*n + random_between(-10,10)*pi/180),
                        radius * random_between(0.8,1.2) * sin(2*pi/(n_agents)*n + random_between(-10,10)*pi/180) ] for n in range(n_agents)]
    goal_position    = [[-x+random_between(-1.5,1.5),-y+random_between(-1.5,1.5)] for x,y in start_positions]
    agent_trajectories = {}
    agent_velocities   = {}
    agents_optimal_input = {} # the input is a velocity change (a.k.a an acceleration in discrete settings)



    agents : list[Agent] = []
    for n in range(n_agents) :
        a = Agent(identifier=n, position=Vector(*start_positions[n]),goal_position= Vector(*goal_position[n]))
        agents.append(a)
        agent_trajectories[ a.identifier] = {}
        agent_velocities  [ a.identifier ] = {}
        agents_optimal_input[a.identifier] = {}
        
    # set up the callbacks to get the position for each agent
    for agent in agents:
        for other_agent in agents:
            if agent.identifier != other_agent.identifier:
                agent.position_callbacks.append(other_agent.get_position_callback)


    simulation_steps = time_samples
    for steps in tqdm(range(simulation_steps)):
        
        
        for agent in agents:
            agent_trajectories[agent.identifier][steps*agent.time_step] = agent.position
            agent_velocities  [agent.identifier][steps*agent.time_step] = agent.velocity
            agent.notify_position()
        
        for agent in agents:
            agent.step()
            agents_optimal_input[agent.identifier][steps*agent.time_step] = agent.optimal_input
        

    dataset_per_agent = prepare_dataset(agents=agents,
                                           agent_trajectories=agent_trajectories,
                                           agent_velocities = agent_velocities,
                                           agents_input = agents_optimal_input)

    if show_figure :
        fig, axs = plt.subplots(2)

        for agent in agents:
            
            x = [pos.x for pos in agent_trajectories[agent.identifier].values()]
            y = [pos.y for pos in agent_trajectories[agent.identifier].values()]
            axs[0].plot(x,y)
            axs[0].scatter(x[-1],y[-1],c="red")
            axs[0].scatter(x[0],y[0],c="blue")
            
            axs[1].imshow(dataset_per_agent[agent.identifier]["feature"].T,aspect='auto')
        
    
    return dataset_per_agent



def simulate_neuromorphic_controller(radius, n_agents,time_samples, controller, which_agents :list[int],show_figure:bool = False) :

    start_positions  = [[radius * random_between(0.8,1.2) * cos(2*pi/(n_agents)*n + random_between(-10,10)*pi/180),
                        radius * random_between(0.8,1.2) * sin(2*pi/(n_agents)*n + random_between(-10,10)*pi/180) ] for n in range(n_agents)]
    goal_position    = [[-x+random_between(-1.5,1.5),-y+random_between(-1.5,1.5)] for x,y in start_positions]
    agent_trajectories = {}
    agent_velocities   = {}
    agents_optimal_input = {} # the input is a velocity change (a.k.a an acceleration in discrete settings)



    agents : list[Agent] = []
    for n in range(n_agents) :
        a = Agent(identifier=n, position=Vector(*start_positions[n]),goal_position= Vector(*goal_position[n]))
        
        if n in which_agents:
            a.set_neuromorphic_controller(controller)
            
        agents.append(a)
        agent_trajectories[ a.identifier] = {}
        agent_velocities  [ a.identifier ] = {}
        agents_optimal_input[a.identifier] = {}
    
    
    
    # set up the callbacks to get the position for each agent
    for agent in agents:
        for other_agent in agents:
            if agent.identifier != other_agent.identifier:
                agent.position_callbacks.append(other_agent.get_position_callback)


    simulation_steps = time_samples
    for steps in tqdm(range(simulation_steps)):
        
        
        for agent in agents:
            agent_trajectories[agent.identifier][steps*agent.time_step] = agent.position
            agent_velocities  [agent.identifier][steps*agent.time_step] = agent.velocity
            agent.notify_position()
        
        for agent in agents:
            agent.step()
            agents_optimal_input[agent.identifier][steps*agent.time_step] = agent.optimal_input
        

    dataset_per_agent = prepare_dataset(agents=agents,
                                           agent_trajectories=agent_trajectories,
                                           agent_velocities = agent_velocities,
                                           agents_input = agents_optimal_input)

    if show_figure :
        fig, axs = plt.subplots(2)

        for agent in agents:
            
            x = [pos.x for pos in agent_trajectories[agent.identifier].values()]
            y = [pos.y for pos in agent_trajectories[agent.identifier].values()]
            axs[0].plot(x,y,label="Agent "+str(agent.identifier))
            axs[0].scatter(x[-1],y[-1],c="red")
            axs[0].scatter(x[0],y[0],c="blue")
            
            
            # axs[1].imshow(dataset_per_agent[agent.identifier]["feature"].T,aspect='auto')
        axs[0].legend()
    
    return dataset_per_agent