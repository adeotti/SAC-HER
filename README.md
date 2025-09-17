Implementation of SAC and HER algorithm on two different environments with different goals. I implemented SAC+HER on the gymnasium Sparse reward environment and only SAC without HER on the Robosuite Dense reward environment.

## Gymnasium environment 
Contain the code for soft actor critic + Her algorithm implemented on [gymnasium robotics pick and place environment](https://robotics.farama.org/envs/fetch/pick_and_place). The base environment produces a sparse binary reward, and that's why I implemented HER(Hindsight Replay Experiement) since it works better for sparse reward environment. I wrote a custom class inheriting from Gym.wrapper to modify some attributes of the base environment, and to be more specific, I've modified the initial position of the robot, the block on the table, and how often the target is on the table and in the air. That last part is important because experiments proved that that pick and place environment is almost if not impossible to solve if the target is only always in the air like it's designed originally in the base gym robotics (HER original paper, footnote page 6)(I did also ran some test but couldn't solve it too when the target is always in the air). I also couldn't find any other way to modify how often the target appears in the air; the base environment doesn't provide any easy entry point like a parameter in a method, for example.

- HER implementation details: 
I'm using a thread-safe Queue to prefetch some training data from the warmup phase to the queue and start pulling data from it to avoid or reduce idle time during training. Using threading here is important from what i saw because sampling the HER batches is computationally expensive, and after some tests, using a thread + queue combo for the data collection sped the training by more than 5X.

## Gobosuite environment 
Contain the code for SAC but on top of the [robosuite Stack environment](https://robosuite.ai/docs/modules/environments.html#block-stacking) using the Panda robot and with three three-finger dexterous gripper. The goal for the Stack environment is to stack two blocks on top of one another. I use a dense reward version of that environment for my implementation. For the second environment, Lift, the goal is the same as the early gym pick and place environment in the gym env folder; the only real difference is the observation space (much larger here than in gym, mostly because the robot is more complex and the gripper is also different)


## Training and Vectorization method (trained on kaggle)
I use sync mode with disabled [autoreset](https://farama.org/Vector-Autoreset-Mode) for both gym and Robosuite environments instead of the async method because, after running many tests, the asynchronous method just doesn't work in the Jupyter environment for both environments. One way to use Async mode is to convert the notebooks to Python, then import that file as a dataset and run the code with Python's built-in exec function. Another, more straightforward method might be to add the path of the Python file to the system using sys. Also, Async vectorization mode does not work in the Robosuite environment; it only works with the Gym environment 

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

Why I disabled [autoreset](https://farama.org/Vector-Autoreset-Mode) for the vectorized gymnasium environment: 

```python 
def vec_env():
    def make_env():
        x = gym.make("FetchPickAndPlace-v3",max_episode_steps=100)
        x = custom_environment(x)
        return x
    return SyncVectorEnv(
        [make_env for _ in range(hypers.num_envs)],autoreset_mode=gym.vector.AutoresetMode.DISABLED
    )
``` 
The reward with auto reset mode set to "next steps" yields 0.0 at the first step of each reset episode, which might lead to some instability during training since the environment gives a positive reward even though it is not solved. A way to avoid that issue is to manually reset the environment.
I didn't disable autorest for the vectorized Robosuite environment because the reward function returns dense rewards.


# References
- [Soft Actor-Critic Original paper 1 (2018)](https://arxiv.org/abs/1801.01290)
- [Soft Actor-Critic Original paper 2 (2019)](https://arxiv.org/abs/1812.05905)
- [Hindsight Experience Replay original paper](https://arxiv.org/abs/1707.01495)
- [original author's implementation (TensorFlow)](https://github.com/haarnoja/sac)
- [Useful SAC Pytorch implementation](https://github.com/pranz24/pytorch-soft-actor-critic)
- [Robosuite benchmark setup](https://github.com/ARISE-Initiative/robosuite-benchmark)
- [OpenAI SAC implementation (Pytorch)](https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/sac)


