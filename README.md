Implementation of Soft Actor-Critic and Hindsight Experience Replay algorithm on two different environments with different goals. I implemented SAC+HER on the Gymnasium sparse reward environment and SAC without HER on the Robosuite dense reward environment.


***Gymnasium environment : SAC + HER ***

Contain the code for soft actor critic + Her algorithm implemented on [gymnasium robotics pick and place environment](https://robotics.farama.org/envs/fetch/pick_and_place). The base environment produces a sparse binary reward, and that's why I implemented HER(Hindsight Replay Experiement) since it works better for sparse reward environment. I wrote a custom class inheriting from Gym.wrapper to modify some attributes of the base environment (the x,y position of the base robot and the z position of the block).

The ratio of HER/normal transition is 4:1 which is proven to be one of the most optimal (They showed that k between 4 and 8 perform best coupled with the "future" strategy).So the strategy used here is "Future" and I'm using a thread-safe Queue to prefetch some training data from the warmup phase to the queue and start pulling data from it to avoid or reduce idle time during training. Using threading here is important from what i saw because sampling the HER batches is computationally expensive, and after some tests, using a thread + queue combo for the data collection sped the training by more than 5X.


***Robosuite environment : SAC***

Contain the code for SAC but on top of the [robosuite Stack environment](https://robosuite.ai/docs/modules/environments.html#block-stacking) using the Panda robot and with three three-finger dexterous gripper. The goal for the Stack environment is to stack two blocks on top of one another. I use a dense reward version of that environment for my implementation. For the second environment, Lift, the goal is the same as the early gym pick and place environment in the gym env folder; the only real difference is the observation space (much larger here than in gym, mostly because the robot is more complex and the gripper is also different)


***Training and Vectorization method (trained on kaggle)***

I use sync mode with disabled [autoreset](https://farama.org/Vector-Autoreset-Mode) for both gym and Robosuite environments instead of the async method because, after running many tests, the asynchronous method just doesn't work in the Jupyter environment for both environments. One way to use Async mode is to convert the notebooks to Python, then import that file as a dataset and run the code with Python's built-in exec function. Another, more straightforward method might be to add the path of the Python file to the system using sys. Also, Async vectorization mode does not work in the Robosuite environment; it only works with the Gym environment.

```python
with open(python file path,"r") as file: # method 1 
    code = file.read()
    exec(code)

import sys # method 2 
sys.path.append(python file folder)
import file_name # importing it will auto launch the training if main().train(True)
 ```

I disabled [autoreset](https://farama.org/Vector-Autoreset-Mode) for the vectorized gymnasium environment because the reward with auto reset mode set to "next steps" yields 0.0 at the first step of each reset episode, which might lead to some instability during training since the environment gives a positive reward even though it is not solved. A way to avoid that issue that i found is to manually reset the done environments. I did not disabled autorest for the vectorized Robosuite environment because the reward function returns dense rewards.


***References***
1. [Soft Actor-Critic paper 1 (2018)](https://arxiv.org/abs/1801.01290)
2. [Soft Actor-Critic paper 2 (2019)](https://arxiv.org/abs/1812.05905)
3. [Hindsight Experience Replay paper](https://arxiv.org/abs/1707.01495)
4. [SAC author's implementation (TensorFlow)](https://github.com/haarnoja/sac)
5. [Useful SAC (Pytorch) implementation](https://github.com/pranz24/pytorch-soft-actor-critic)
6. [Robosuite benchmark setup](https://github.com/ARISE-Initiative/robosuite-benchmark)
7. [OpenAI SAC implementation (Pytorch)](https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/sac)


