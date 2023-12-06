# CSCE 642: Deep Reinforcement Learning - Final Project

Connor Temple, Praneeth Somisetty

TD3 and DDPG implementations for image-based autonomous driving in TrackMania. 
This project was built using the [TMRL](https://github.com/trackmania-rl/tmrl) framework, and code was modified from TMRL's competition [script](https://github.com/trackmania-rl/tmrl/blob/master/tmrl/tuto/competition/custom_actor_module.py) using OpenAI's SpinningUp implementations of [DDPG](https://spinningup.openai.com/en/latest/algorithms/ddpg.html) and [TD3](https://spinningup.openai.com/en/latest/algorithms/td3.html).
## System Requirements (Provided in TMRL [Installation Instructions](https://github.com/trackmania-rl/tmrl/blob/master/readme/Install.md))
- Windows (required for running TrackMania)
- Python >= 3.7
- "recent" NVIDIA GPU

## Setup
The TMRL github provides specific instructions for installing and setting up the environment, which can be found [here](https://github.com/trackmania-rl/tmrl/blob/master/readme/Install.md). Information about testing the environment setup can be found [here](https://github.com/trackmania-rl/tmrl/blob/master/readme/get_started.md). 

### Note:
To use our custom track, when copying the `tmrl-test.Map.Gbx` file into the directory specified in the setup [instructions](https://github.com/trackmania-rl/tmrl/blob/master/readme/get_started.md) mentioned above, be sure to copy `TMRL_Test_easy.Map.Gbx` found in the `custom_params` folder as well:

- Navigate to the `thisprojectname\custom_params` folder
- Copy the `TMRL_Test_easy.Map.Gbx` file into `...\Documents\Trackmania\Maps\My Maps` (or equivalent location on your system).


## Using our Implementation
The tutorial contained in the competition [script](https://github.com/trackmania-rl/tmrl/blob/master/tmrl/tuto/competition/custom_actor_module.py) provides steps to run custom instances. As our code is a modified version of this script, the same steps are taken. 

### To use our custom track: 
To train using our custom track, reference the note above. Provided in the `custom_params` folder is a copy of the reward we set for our custom track, `reward.pkl`. 

- Navigate to `\thisprojectname\custom_params`
- Copy the `reward.pkl` file into the `...\TmrlData\reward` folder, say "yes" to replace the existing `reward.pkl` file. 
    - A copy of tmrl's reward can be found in the `TmrlData\resources` folder, so if you every want to use their reward on their test track, you can find it there.

If you would rather use the track provided by tmrl (and its respective reward), skip the above steps.  

### Using our `config_XXX.json` files
Also found in the `custom_params` folder are copies of our different `config.json` files, labeled with their respective algorithms. Each "RUN_NAME" has been kept original to our experiments, but the "WANDB" project is changed. You can feel free to change the names and wandb information as you see fit, but each config file will send data to the project listed. WandB proved to be a very useful tool for collecting data about our runs. 

To use these `config_XXX.json` files:

- Navigate to the `\thisprojectname\custom_params` folder
- Copy the desired `config_XXX.json` file into the `...\TmrlData\config` folder.
- Navigate to the `...\TmrlData\config` folder, and delete the original `config.json` file.
    - A copy of the original can be found in the `...\TmrlData\resources` folder.
- Rename our `config_XXX.json` file to `config.json`.

In reality, you can do this however you deem fit. These instructions are just one way to replace the config file. 

**Note**: If you do plan to train an entire model, I would recommend performing the `config` switch or at least changing the WANDB parameters, as the default provided by TMRL sends data to their WANDB instance. 

### To train:
Provided the setup was done correctly, training should be fairly straightforward
Then, as the tutorial script suggests, open three different terminals in the location of our custom scripts and run (using td3 as an example):

```shell
python custom_td3.py --server
```
```shell
python custom_td3.py --trainer
```
```shell
python custom_td3.py --worker
```

Replace `custom_td3.py` with either `custom_ddpg.py` for DDPG or `custom_actor_module.py` for TMRL's SAC.

After running the `trainer` and `worker`, quickly click on the windowed TrackMania instance, then watch as the model trains!

### Note:
As you may have read in tmrl's documentation, training on a single device can be very computationally intesive. We were able to train our three instances of the algorithms with specs as follows:
- Ryzen 5 5600X CPU
- 16GB 3200MHz DDR4 CL16 RAM
- NVIDIA RTX 3070 8GB LHR GPU