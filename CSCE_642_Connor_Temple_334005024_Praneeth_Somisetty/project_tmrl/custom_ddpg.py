"""
==============================================================================
COMPETITION TUTORIAL #1: Custom model and RL algorithm
==============================================================================

In this tutorial, we will customize the TrackMania training pipeline.

The tutorial works with the TrackMania FULL Gymnasium environment.
Please refer to the README on GitHub to set up this environment in config.json:
https://github.com/trackmania-rl/tmrl#full-environment

Note: This tutorial describes implementing and running a TrainingAgent along with an ActorModule.
It is relevant if you want to implement your own RL approaches in TrackMania.
If you plan to try non-RL approaches instead, this is also accepted:
just use the competition Gymnasium Full environment and do whatever you need,
then, wrap your trained policy in an ActorModule, and submit your entry :)

Copy and adapt this script to implement your own algorithm/model in TrackMania.
Then, use the script as follows:

To launch the Server, provided the script is named custom_actor_module.py, execute:
python custom_actor_module.py --server

In another terminal, launch the Trainer:
python custom_actor_module.py --trainer

And in yet another terminal, launch a RolloutWorker:
python custom_actor_module.py --worker

You can launch these in any order, but we recommend server, then trainer, then worker.
If you are running everything on the same machine, your trainer may consume all your resource,
resulting in your worker struggling to collect samples in a timely fashion.
If your worker crazily warns you about time-steps timing out, this is probably the issue.
The best way of using TMRL with TrackMania is to have your worker(s) and trainer on separate machines.
The server can run on either of these machines, or yet another machine that both can reach via network.
Achieving this is easy (and is also kind of the whole point of the TMRL framework).
Just adapt config.json (or this script) to your network configuration.
In particular, you will want to set the following in the TMRL config.json file of all your machines:

"LOCALHOST_WORKER": false,
"LOCALHOST_TRAINER": false,
"PUBLIC_IP_SERVER": "<ip.of.the.server>",
"PORT": <port of the server (usually requires port forwarding if accessed via the Internet)>,

If you are training over the Internet, please read the security instructions on the TMRL GitHub page.
"""

"""
- Make changes to DDPG to make work
    - Add a custom instance of RolloutWorker and/or TrainingOffline to call a new function for selecting actions
        - will make the process of distinguishing a "random" sample and the noisy deterministic policy. 
    - Ensure our current network structure will work with the slight change to MLP (I think it should be fine)
    - https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/ddpg 
- Make a new file for TD3, and make it work correctly
    - https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/td3 
"""

# The constants that are defined in config.json:
import tmrl.config.config_constants as cfg
# Useful classes:
import tmrl.config.config_objects as cfg_obj
# The utility that TMRL uses to partially instantiate classes:
from tmrl.util import partial
# The TMRL three main entities (i.e., the Trainer, the RolloutWorker and the central Server):
from tmrl.networking import Trainer, RolloutWorker, Server

# The training class that we will customize with our own training algorithm in this tutorial:
from tmrl.training_offline import TrainingOffline

# And a couple external libraries:
import numpy as np
import os
import datetime


# =====================================================================
# USEFUL PARAMETERS
# =====================================================================

# Maximum number of training 'epochs':
# (training is checkpointed at the end of each 'epoch', this is also when training metrics can be logged to wandb)
epochs = cfg.TMRL_CONFIG["MAX_EPOCHS"]

# Number of rounds per 'epoch':
# (training metrics are displayed in the terminal at the end of each round)
rounds = cfg.TMRL_CONFIG["ROUNDS_PER_EPOCH"]

# Number of training steps per round:
# (a training step is a call to the train() function that we will define later in this tutorial)
steps = cfg.TMRL_CONFIG["TRAINING_STEPS_PER_ROUND"]

# Minimum number of environment steps collected before training starts:
# (this is useful when you want to fill your replay buffer with samples from a baseline policy)
start_training = cfg.TMRL_CONFIG["ENVIRONMENT_STEPS_BEFORE_TRAINING"]

# Maximum training steps / environment steps ratio:
# (if training becomes faster than this ratio, it will be paused, waiting for new samples from the environment)
max_training_steps_per_env_step = cfg.TMRL_CONFIG["MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP"]

# Number of training steps performed between broadcasts of policy updates:
update_model_interval = cfg.TMRL_CONFIG["UPDATE_MODEL_INTERVAL"]

# Number of training steps performed between retrievals of received samples to put them in the replay buffer:
update_buffer_interval = cfg.TMRL_CONFIG["UPDATE_BUFFER_INTERVAL"]

# Training device (e.g., "cuda:0"):
device_trainer = 'cuda' if cfg.CUDA_TRAINING else 'cpu'

# Maximum size of the replay buffer:
memory_size = cfg.TMRL_CONFIG["MEMORY_SIZE"]

# Batch size for training:
batch_size = cfg.TMRL_CONFIG["BATCH_SIZE"]

# Wandb credentials:
wandb_run_id = cfg.WANDB_RUN_ID  
wandb_project = cfg.TMRL_CONFIG["WANDB_PROJECT"]  # name of the wandb project in which your run will appear
wandb_entity = cfg.TMRL_CONFIG["WANDB_ENTITY"]  # wandb account
wandb_key = cfg.TMRL_CONFIG["WANDB_KEY"]  # wandb API key

os.environ['WANDB_API_KEY'] = wandb_key  # sets wandb API key as the active key

# Number of time-steps after which episodes collected by the worker are truncated:
max_samples_per_episode = cfg.TMRL_CONFIG["RW_MAX_SAMPLES_PER_EPISODE"]

# Networking parameters:
# (In TMRL, networking is managed by tlspyo. The following are tlspyo parameters.)
server_ip_for_trainer = cfg.SERVER_IP_FOR_TRAINER  # IP of the machine running the Server (trainer point of view)
server_ip_for_worker = cfg.SERVER_IP_FOR_WORKER  # IP of the machine running the Server (worker point of view)
server_port = cfg.PORT  # port used to communicate with this machine
password = cfg.PASSWORD  # password that secures your communication
security = cfg.SECURITY  # when training over the Internet, it is safer to change this to "TLS"
# (please read the security instructions on GitHub)


# =====================================================================
# ADVANCED PARAMETERS
# =====================================================================

# Base class of the replay memory used by the trainer:
memory_base_cls = cfg_obj.MEM

# Sample compression scheme applied by the worker for this replay memory:
sample_compressor = cfg_obj.SAMPLE_COMPRESSOR

# Sample preprocessor for data augmentation:
sample_preprocessor = None

# Path from where an offline dataset can be loaded to initialize the replay memory:
dataset_path = cfg.DATASET_PATH

# Preprocessor applied by the worker to the observations it collects:
# (Note: if your script defines the name "obs_preprocessor", we will use your preprocessor instead of the default)
obs_preprocessor = cfg_obj.OBS_PREPROCESSOR


# =====================================================================
# COMPETITION FIXED PARAMETERS (Do Not Change!)
# =====================================================================

# rtgym environment class (full TrackMania Gymnasium environment):
env_cls = cfg_obj.ENV_CLS

# Device used for inference on workers (change if you like but keep in mind that the competition evaluation is on CPU)
device_worker = 'cpu'


# =====================================================================
# ENVIRONMENT PARAMETERS
# =====================================================================

# Dimensions of the TrackMania window:
window_width = cfg.WINDOW_WIDTH  # must be between 256 and 958
window_height = cfg.WINDOW_HEIGHT  # must be between 128 and 488

# Dimensions of the actual images in observations:
img_width = cfg.IMG_WIDTH
img_height = cfg.IMG_HEIGHT

# Whether you are using grayscale (default) or color images:
img_grayscale = cfg.GRAYSCALE

# Number of consecutive screenshots in each observation:
imgs_buf_len = cfg.IMG_HIST_LEN

# Number of actions in the action buffer (this is part of observations):
act_buf_len = cfg.ACT_BUF_LEN


# =====================================================================
# MEMORY CLASS
# =====================================================================

memory_cls = partial(memory_base_cls,
                     memory_size=memory_size,
                     batch_size=batch_size,
                     sample_preprocessor=sample_preprocessor,
                     dataset_path=cfg.DATASET_PATH,
                     imgs_obs=imgs_buf_len,
                     act_buf_len=act_buf_len,
                     crc_debug=False)


# =====================================================================
# CUSTOM MODEL
# =====================================================================

from tmrl.actor import TorchActorModule
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from math import floor


# In the full version of the TrackMania 2020 environment, the
# observation-space comprises a history of screenshots. Thus, we need
# Computer Vision layers such as a CNN in our model to process these.
# The observation space also comprises single floats representing speed,
# rpm and gear. We will merge these with the information contained in
# screenshots thanks to an MLP following our CNN layers.


# Here is the MLP:
def mlp(sizes, activation, output_activation=nn.Identity):
    """
    A simple MLP (MultiLayer Perceptron).

    Args:
        sizes: list of integers representing the hidden size of each layer
        activation: activation function of hidden layers
        output_activation: activation function of the last layer

    Returns:
        Our MLP in the form of a Pytorch Sequential module
    """
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


# The next utility computes the dimensionality of CNN feature maps when flattened together:
def num_flat_features(x):
    size = x.size()[1:]  # dimension 0 is the batch dimension, so it is ignored
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


# The next utility computes the dimensionality of the output in a 2D CNN layer:
def conv2d_out_dims(conv_layer, h_in, w_in):
    h_out = floor((h_in + 2 * conv_layer.padding[0] - conv_layer.dilation[0] * (conv_layer.kernel_size[0] - 1) - 1) / conv_layer.stride[0] + 1)
    w_out = floor((w_in + 2 * conv_layer.padding[1] - conv_layer.dilation[1] * (conv_layer.kernel_size[1] - 1) - 1) / conv_layer.stride[1] + 1)
    return h_out, w_out


class MyCNN(nn.Module):
    def __init__(self, q_net):
        """
        Simple CNN (Convolutional Neural Network) model for DDPG & TD3.

        Args:
            q_net (bool): indicates whether this neural net is a critic network

        CSCE 642 Notes:
            The main difference between this and TMRL's VanillaCNN is an additional MLP layer with TanH smoothing.
            More details where the MLP is defined below. 
        """
        super(MyCNN, self).__init__()

        self.q_net = q_net

        # Convolutional layers processing screenshots:
        # The default config.json gives 4 grayscale images of 64 x 64 pixels
        self.h_out, self.w_out = img_height, img_width
        self.conv1 = nn.Conv2d(imgs_buf_len, 64, 8, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv1, self.h_out, self.w_out)
        self.conv2 = nn.Conv2d(64, 64, 4, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv2, self.h_out, self.w_out)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv3, self.h_out, self.w_out)
        self.conv4 = nn.Conv2d(128, 128, 4, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv4, self.h_out, self.w_out)
        self.out_channels = self.conv4.out_channels

        # Dimensionality of the CNN output:
        self.flat_features = self.out_channels * self.h_out * self.w_out

        # Dimensionality of the MLP input:
        # The MLP input will be formed of:
        # - the flattened CNN output
        # - the current speed, gear and RPM measurements (3 floats)
        # - the 2 previous actions (2 x 3 floats), important because of the real-time nature of our controller
        # - when the module is the critic, the selected action (3 floats)
        float_features = 12 if self.q_net else 9
        self.mlp_input_features = self.flat_features + float_features

        # MLP layers:
        # Instead of sampling from a multivariate gaussian like SAC, we are taking only the deterministic action based
        # on our policy network, so we added a third layer to our non-q_net mlp to return our action. Since we get our
        # action now instead of later, we also take the Tanh here. 
        self.mlp_layers = [256, 256, 1] if self.q_net else [256, 256, 3]
        if self.q_net:
            self.mlp = mlp([self.mlp_input_features] + self.mlp_layers, nn.ReLU)
        else: 
            self.mlp = mlp([self.mlp_input_features] + self.mlp_layers, nn.ReLU, nn.Tanh) # Additional TanH smoothing

    def forward(self, x):
        """
        Forward step through our network

        Args:
            x (torch.Tensor): input tensor (i.e., the observation fed to our deep neural network)

        Returns:
            the output of our neural network in the form of a torch.Tensor 
            Either a Q value, or an Action.

        CSCE 642 Notes:
            This is unchanged from TMRL's implementation, as the main point is to process images, and our changes
            work with this version. 
        """
        if self.q_net:
            # The critic takes the next action (act) as additional input
            # act1 and act2 are the actions in the action buffer (real-time RL):
            speed, gear, rpm, images, act1, act2, act = x
        else:
            # For the policy, the next action (act) is what we are computing, so we don't have it:
            speed, gear, rpm, images, act1, act2 = x

        # Forward pass of our images in the CNN:
        # (note that the competition environment outputs histories of 4 images
        # and the default config outputs these as 64 x 64 greyscale,
        # we will stack these greyscale images along the channel dimension of our input tensor)
        x = F.relu(self.conv1(images))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # Now we will flatten our output feature map.
        # Let us double-check that our dimensions are what we expect them to be:
        flat_features = num_flat_features(x)
        assert flat_features == self.flat_features, f"x.shape:{x.shape},\
                                                    flat_features:{flat_features},\
                                                    self.out_channels:{self.out_channels},\
                                                    self.h_out:{self.h_out},\
                                                    self.w_out:{self.w_out}"
        # All good, let us flatten our output feature map:
        x = x.view(-1, flat_features)

        # Finally, we can feed the result along our float values to the MLP:
        if self.q_net:
            x = torch.cat((speed, gear, rpm, x, act1, act2, act), -1)
        else:
            x = torch.cat((speed, gear, rpm, x, act1, act2), -1)
        x = self.mlp(x)

        # And this gives us the output of our deep neural network :)
        return x

# ==============================================  TMRL  =============================================================
# During training, TMRL will regularly save our trained ActorModule in the TmrlData/weights folder.
# By default, this would be done using the torch (i.e., pickle) serializer.
# However, while saving and loading your own pickle files is fine,
# it is highly dangerous to load other people's pickle files.
# Therefore, the competition submission does not accept pickle files.
# Instead, we can submit our trained weights in the form of a human-readable JSON file.
# The ActorModule interface defines save() and load() methods that we will override with our own JSON serializer.
# ==============================================  TMRL  =============================================================

import json


class TorchJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for torch tensors, used in the custom save() method of our ActorModule.
    """
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().detach().numpy().tolist()
        return json.JSONEncoder.default(self, obj)


class TorchJSONDecoder(json.JSONDecoder):
    """
    Custom JSON decoder for torch tensors, used in the custom load() method of our ActorModule.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, dct):
        for key in dct.keys():
            if isinstance(dct[key], list):
                dct[key] = torch.Tensor(dct[key])
        return dct


class OtherActorModule(TorchActorModule):
    """
    Our policy wrapped in the TMRL ActorModule class.

    TD3 & DDPG allow us to simplify our ActorModule's forward implementation. 

    Code modified based on OpenAI's SpinningUp version of DDPG/TD3. 

    (Note: TorchActorModule is a subclass of ActorModule and torch.nn.Module)
    """
    def __init__(self, observation_space, action_space):
        """
        When implementing __init__, we need to take the observation_space and action_space arguments.

        Args:
            observation_space: observation space of the Gymnasium environment
            action_space: action space of the Gymnasium environment
        """
        super().__init__(observation_space, action_space)

        dim_act = action_space.shape[0] # dimensionality of actions
        act_limit = action_space.high[0] # Maximum value for actions
        # Our policy
        self.net = MyCNN(q_net=False)
        # Save the action limit
        self.act_limit = act_limit
        
    
    def save(self, path):
        """
        JSON-serialize a detached copy of the ActorModule and save it in path.

        IMPORTANT: FOR THE COMPETITION, WE ONLY ACCEPT JSON AND PYTHON FILES.
        IN PARTICULAR, WE *DO NOT* ACCEPT PICKLE FILES (such as output by torch.save()...).

        All your submitted files must be human-readable, for everyone's safety.
        Indeed, untrusted pickle files are an open door for hackers.

        Args:
            path: pathlib.Path: path to where the object will be stored.
        """
        with open(path, 'w') as json_file:
            json.dump(self.state_dict(), json_file, cls=TorchJSONEncoder)
        # torch.save(self.state_dict(), path)

    def load(self, path, device):
        """
        Load the parameters of your trained ActorModule from a JSON file.

        Adapt this method to your submission so that we can load your trained ActorModule.

        Args:
            path: pathlib.Path: full path of the JSON file
            device: str: device on which the ActorModule should live (e.g., "cpu")

        Returns:
            The loaded ActorModule instance
        """
        self.device = device
        with open(path, 'r') as json_file:
            state_dict = json.load(json_file, cls=TorchJSONDecoder)
        self.load_state_dict(state_dict)
        self.to_device(device)
        # self.load_state_dict(torch.load(path, map_location=self.device))

        return self

    def forward(self, obs):
        """
        Computes the action from an observation by passing through the network.

        Since the network handles the Tanh smoothing, and we only care about the deterministic action, we
        only scale the network output by the action limit and return (while squeezing dimensions).

        Args:
            obs (object): the input observation from the environment.

        Returns:
            action: the computed action, in the form of a torch tensor of 3 values between -1.0 and 1.0

        CSCE 642 Notes:
            This code was modified using OpenAI's SpinningUp implementation of DDPG.
        """
        # Get the deterministic action, and squash it into the action space
        action = self.act_limit * self.net(obs)
        return action.squeeze() # Return, removing the batch dimension
    
    def act(self, obs, test=False):
        """
        Computes an action from an observation. Used by the worker specifically.

        This was slightly more important in a previous version of DDPG, where I attempted to 
        handle the process of adding noise using this. I have since delegated the handling of 
        epsilon to MyRolloutWorker.

        Args:
            obs (object): the input observation (when using TorchActorModule, this is a torch.Tensor)
            test (bool): True at test-time (e.g., during evaluation...), False otherwise

        Returns:
            act (numpy.array): the computed action, in the form of a numpy array of 3 values between -1.0 and 1.0
        """
        with torch.no_grad():
            a = self.forward(obs=obs)
            
            return a.cpu().numpy()

class MyRolloutWorker(RolloutWorker):
    """
    Custom RolloutWorker used for DDPG & TD3.

    RolloutWorker is a class contained in the TMRL library. 

    This was modified to facilitate adding noise and allow for random actions prior to the beginning of training.
    This was done to encourage more exploration early on to help training. 
    """
    def __init__(self, env_cls,
                        actor_module_cls,
                           sample_compressor=sample_compressor,
                           device=device_worker,
                           server_ip=server_ip_for_worker,
                           server_port=server_port,
                           password=password,
                           security=security,
                           max_samples_per_episode=max_samples_per_episode,
                           obs_preprocessor=obs_preprocessor,
                           standalone=False):
        
        super().__init__(env_cls=env_cls,
                           actor_module_cls=actor_module_cls,
                           sample_compressor=sample_compressor,
                           device=device_worker,
                           server_ip=server_ip_for_worker,
                           server_port=server_port,
                           password=password,
                           security=security,
                           max_samples_per_episode=max_samples_per_episode,
                           obs_preprocessor=obs_preprocessor,
                           standalone=standalone)
        
        self.total_steps = 5001 # variable to track when to take DDPG actions
        self.epsilon = 0 # amount of noise to add to actions
        self.act_limit = self.env.action_space.high[0] # Maximum value for actions
        self.dim_act = self.env.action_space.shape[0] # dimension of the action space

        torch.manual_seed(42)
        np.random.seed(42)

    def act(self, obs, test=False):
        """
        Class to call our Custom Actor's "act" function. Adds noise if training

        Args:
            obs: Observaion gathered from the environment. 
            test: Whether we are testing or not

        Returns:
            action: either the action directly from the policy, or action with added noise.
        """
        # Get the action passed through our network
        action = self.actor.act_(obs, test=test)

        if test: # Return the action without noise if testing
            return action
        
        # Add noise to the action for exploration
        action += self.epsilon * np.random.randn(self.dim_act)
        action = np.clip(action, -self.act_limit, self.act_limit) 

        return action


    def step(self, obs, test, collect_samples, last_step=False):
        """
        Class to take a step in the environment. 

        Modified to allow for random sampling prior to training. 

        Called elsewhere, and ultimately we do not really care what else happens here.
        """
        if test or self.total_steps > start_training:
            act = self.act(obs, test=test)
        else:
            act = self.env.action_space.sample() 
            self.total_steps += 1
        
        new_obs, rew, terminated, truncated, info = self.env.step(act)
        
        if self.obs_preprocessor is not None:
            new_obs = self.obs_preprocessor(new_obs)
        if collect_samples:
            if last_step and not terminated:
                truncated = True
            if self.crc_debug:
                info['crc_sample'] = (obs, act, new_obs, rew, terminated, truncated)
            if self.get_local_buffer_sample:
                sample = self.get_local_buffer_sample(act, new_obs, rew, terminated, truncated, info)
            else:
                sample = act, new_obs, rew, terminated, truncated, info
            self.buffer.append_sample(sample)  # CAUTION: in the buffer, act is for the PREVIOUS transition (act, obs(act))
        return new_obs, rew, terminated, truncated, info


class MyCNNQFunction(nn.Module):
    """
    Critic module for DDPG.

    This remains unchanged, as the functionality is the same. 
    """
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.net = MyCNN(q_net=True)  # q_net is True for a critic module

    def forward(self, obs, act):
        """
        Estimates the action-value of the (obs, act) state-action pair.

        In RL theory, the action-value is the expected sum of (gamma-discounted) future rewards
        when observing obs, taking action act, and following the current policy ever after.

        Args:
            obs: current observation
            act: tried next action

        Returns:
            The action-value of act in situation obs, as estimated by our critic network
        """
        # Since q_net is True, we append our action act to our observation obs.
        x = (*obs, act)
        q = self.net(x)
        return torch.squeeze(q, -1)



class DDPGCNNActorCritic(nn.Module):
    """
    Actor-critic module for the DDPG algorithm.
    """
    def __init__(self, observation_space, action_space):
        super().__init__()
        # Policy Network
        self.actor = OtherActorModule(observation_space, action_space)
        # Value Network
        self.q = MyCNNQFunction(observation_space, action_space)

# =====================================================================
# CUSTOM TRAINING ALGORITHM
# =====================================================================

from tmrl.training import TrainingAgent
from tmrl.custom.utils.nn import copy_shared, no_grad
from tmrl.util import cached_property
from copy import deepcopy
import itertools
from torch.optim import Adam

# We implement the DDPG algorithm by adapting the OpenAI Spinup implementation.

class DDPGTrainingAgent(TrainingAgent):
    """
    Our custom training algorithm for DDPG.
    """
    model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model)))

    def __init__(self,
                 observation_space=None, # Gymnasium observation space (required argument here for your convenience)
                 action_space=None, # Gymnasium action space (required argument here for your convenience)
                 device=None, # Device our TrainingAgent should use for training (required argument)
                 model_cls=DDPGCNNActorCritic, # An actor-critic module, encapsulating our ActorModule
                 gamma=0.99, # Discount factor
                 polyak=0.995, # Exponential averaging factor for the target actor/critic
                 lr_actor=1e-3, # Learning rate for the actor
                 lr_critic=1e-3): # Learning rate for the critic
        super().__init__(observation_space=observation_space,
                         action_space=action_space,
                         device=device)
        
        model = model_cls(observation_space, action_space)
        self.model = model.to(self.device)
        self.model_target = no_grad(deepcopy(self.model))
        self.gamma = gamma
        self.polyak = polyak
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.pi_optimizer = Adam(self.model.actor.parameters(), lr=self.lr_actor)
        self.q_optimizer = Adam(self.model.q.parameters(), lr=lr_critic)

    def get_actor(self):
        """
        Returns a copy of the current ActorModule.

        We return a copy without gradients, as this is for sending to the RolloutWorkers.

        Returns:
            actor: ActorModule: updated actor module to forward to the worker(s)
        """
        return self.model_nograd.actor
    
    def train(self, batch):
        """
        Executes a training iteration from batched training samples (batches of RL transitions).

        A training sample is of the form (o, a, r, o2, d, t) where:
        -> o is the initial observation of the transition
        -> a is the selected action during the transition
        -> r is the reward of the transition
        -> o2 is the final observation of the transition
        -> d is the "terminated" signal indicating whether o2 is a terminal state
        -> t is the "truncated" signal indicating whether the episode has been truncated by a time-limit

        Note that we will ignore the t signal for a clever reason.
        When the episode is truncated due to a time limit, we do not want our model to believe that o2 is terminal.
        Instead, we will make our model believe that the episode would normally have continued after this transition.
        Because we use a discount factor, this does not result in an infinite value.
        In our situation, the discount factor is what incentivizes the AI to run as fast as possible!

        Args:
            batch: (previous observation, action, reward, new observation, terminated signal, truncated signal)

        Returns:
            logs: Dictionary: a python dictionary of training metrics you wish to log on wandb
        """
        # Split batch into relevant components
        o, a, r, o2, d, _ = batch
        # Sample an action in the current policy (not entirely sure why tmrl does this here, but they do so we kept it).
        pi = self.model.actor(o)
        # Compute Q estimates for current observation
        q = self.model.q(o, a)
        # Compute target Q value using our target actor
        with torch.no_grad():
            q_pi_targ = self.model_target.q(o2,self.model_target.actor(o2))
            backup = r + self.gamma * (1 - d) * q_pi_targ
        # Critic loss
        loss_q = ((q - backup)**2).mean()
        # Train the critic network
        self.q_optimizer.zero_grad()
        loss_q.backward()
        self.q_optimizer.step()
        # Policy optimization step
        for p in self.model.q.parameters():
            p.requires_grad = False
        
        q_pi = self.model.q(o, pi)
        # Negative mean for Gradient Ascent
        loss_pi = -q_pi.mean()
        # Perform Gradient Ascent step to train policy
        self.pi_optimizer.zero_grad()
        loss_pi.backward()
        self.pi_optimizer.step()

        for p in self.model.q.parameters():
            p.requires_grad = True
        # Update the target model
        with torch.no_grad():
            for p, p_targ in zip(self.model.parameters(), self.model_target.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
        
        ret_dict = dict(
            loss_actor=loss_pi.detach().item(),
            loss_critic=loss_q.detach().item(),
        )
        return ret_dict
            

# "partially" instantiate our training agent. This is a TMRL specific functionality. 

training_agent_cls = partial(DDPGTrainingAgent,
                             model_cls=DDPGCNNActorCritic,
                             gamma=0.99,
                             polyak=0.995,
                             lr_actor=0.00005,
                             lr_critic=0.00003) 


# =====================================================================
# TMRL TRAINER
# =====================================================================

# Partially instantiate the training class with our custom classes and global parameters.

training_cls = partial(
    TrainingOffline,
    env_cls=env_cls,
    memory_cls=memory_cls,
    training_agent_cls=training_agent_cls,
    epochs=epochs,
    rounds=rounds,
    steps=steps,
    update_buffer_interval=update_buffer_interval,
    update_model_interval=update_model_interval,
    max_training_steps_per_env_step=max_training_steps_per_env_step,
    start_training=start_training,
    device=device_trainer)


# =====================================================================
# RUN YOUR TRAINING PIPELINE
# =====================================================================
# The training pipeline configured in this tutorial runs with the "TM20FULL" environment.

# You can configure the "TM20FULL" environment by following the instruction on GitHub:
# https://github.com/trackmania-rl/tmrl#full-environment

# In TMRL, a training pipeline is made of
# - one Trainer (encompassing the training algorithm that we have coded in this tutorial)
# - one to several RolloutWorker(s) (encompassing our ActorModule and the Gymnasium environment of the competition)
# - one central Server (through which RolloutWorker(s) and Trainer communicate)
# Let us instantiate these via an argument that we will pass when calling this script:

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--server', action='store_true', help='launches the server')
    parser.add_argument('--trainer', action='store_true', help='launches the trainer')
    parser.add_argument('--worker', action='store_true', help='launches a rollout worker')
    parser.add_argument('--test', action='store_true', help='launches a rollout worker in standalone mode')
    args = parser.parse_args()

    if args.trainer:
        my_trainer = Trainer(training_cls=training_cls,
                             server_ip=server_ip_for_trainer,
                             server_port=server_port,
                             password=password,
                             security=security)
        my_trainer.run_with_wandb(entity=wandb_entity,
                                  project=wandb_project,
                                  run_id=wandb_run_id)
    elif args.worker or args.test:
        rw = MyRolloutWorker(env_cls=env_cls,
                           actor_module_cls=OtherActorModule,
                           sample_compressor=sample_compressor,
                           device=device_worker,
                           server_ip=server_ip_for_worker,
                           server_port=server_port,
                           password=password,
                           security=security,
                           max_samples_per_episode=max_samples_per_episode,
                           obs_preprocessor=obs_preprocessor,
                           standalone=args.test)
        rw.run()
    elif args.server:
        import time
        serv = Server(port=server_port,
                      password=password,
                      security=security)
        while True:
            time.sleep(1.0)