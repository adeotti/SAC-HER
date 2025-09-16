Implementation of SAC and HER reward on two distinc environment with differents goals.
I implemented SAC+HER on the gymnasium Sparse reward environment and only SAC without HER on the robosuite Dense reward environment

# gymnasium environment : 
Contain the code for soft actor critic + Her algorithm implemented on [gymnasium robotics pick and place environment](https://robotics.farama.org/envs/fetch/pick_and_place). The base enviornment return a sparse binary reward and that's why I implemented HER(Hindsight Replay Experiement) since it works better for sparse reward environment. I wrote a custom class inheriting from Gym.wrapper to modify some attribute of the base environment and to be more specific, I've modified the initial position of the robot, the block on the table and how often the target is on the table and in the air. That last part is important because experiments proved that that pick and place environment is almost if not impossible to solve if the target is only always in the air like it's designed originally in the base gym robotics (HER original paper,footnote page 6)(I did also ran some test but couldn't solve it too when the target is always in the air). I also couldn't find any other way to modify how often the target appear in the air, the base environment doesn't provide any easy entry proint like a parameters in a method for example.

-  HER implementation details: 
Im using a thread safe Queue to prefetch the some training data from warmup phase to the queue and start pulling data from the Queue to avoid or reduce iddle time during training.Using threading her is important because sampling the HER batch are computationaly expensive and after some test, using a thread + queue combo for the data collection speedup the training by more than 5X.


# robosuite environment : 
Contain the code for sac but on top of the [robosuite Stack environment](https://robosuite.ai/docs/modules/environments.html) using the Panda robot and with three dexterous gripper. The goal for the Stack environment is to stack two block on top of one another, I use a dense reward version of that enviornment for my implementation. For the second environment, Lift, the goal is the same as the early gym pick and place environment in the gym env folder, the only really difference is the observation space (much larger here than in gym mostly because the robot is more complex and the gripper is also different)


# training and Vectorization method (training on kaggle):
I use sync mode with disabled autoreset* for both gym and robosuite environment instead of async methode because after running many test, the asynchronous method just don't work in the jupyter environment for both environment. One way to use Async mode is to convert the nobooks as python then import that file as a dataset and run the code with python's built in exec function or another more straightforward method might be to add the path of the python file to the system using sys. Also, Async vectorization mode do not work in the robosuite environment, it only works with the gym env 

```python
# method 1 
with open(python file path,"r") as file:
    code = file.read()
    exec(code)

# method 2 
import sys
sys.path.append(python file folder)
import file_name # importing it will auto launch the training if main().train(True)
 ```

Why I disabled autoreset[^1] for gym vectorized environment: 

```python 
def vec_env():
    def make_env():
        x = gym.make("FetchPickAndPlace-v3",max_episode_steps=50)
        x = custom_environment(x)
        return x
    return SyncVectorEnv(
        [make_env for _ in range(hypers.num_envs)],autoreset_mode=gym.vector.AutoresetMode.DISABLED
    )
``` 
The reward with auto reset mode set to "next steps" yield 0.0 at the first step of each reseted episodes, that might lead to some instability during training since the environment gives a positive reward even though it is not solved. A way to avoid that issue is to manually reset the environment.
I didn't disabled autorest for the vectorized robosuite environment because the reward function returns denses rewards.


Chack also the reference file for the references that helped me with this algorithm.


# References
- [Soft Actor-Critic Original paper](https://arxiv.org/abs/1812.05905)
- [Hindsight Experience Replay original paper](https://arxiv.org/abs/1707.01495)
- [original author's implementation (TensorFlow)](https://github.com/haarnoja/sac)
- [SAC Pytorch implementation](https://github.com/pranz24/pytorch-soft-actor-critic)
- [Robosuite benchmark setup](https://github.com/ARISE-Initiative/robosuite-benchmark)
- [OpenAI SAC implementation](https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/sac)

[^1]: [Gym Autoreset article for vectorized environments](https://farama.org/Vector-Autoreset-Mode)
