# Reinforcement Learning(Q-learning / DQN / A2C) on the Mountain Car problem

This repository demonstrates the outcome of *Reinforcement Learning (RL)* techniques, namely **Q-learning, Deep Q-Network(DQN), Advantage Actor-Critic(A2C)**, when applied to the **Mountain Car environment** after selecting initial parameters. First, an overview of the essentials of Reinforcement learning, then some background on the Mountain Car problem, RL algorithms & their components followed by the algorithm implementation results.

The code for the experiments involving all the algorithms and their outcomes is available in a [Jupyter notebook](https://github.com/bragancas/texttest/blob/master/DLO_SB_AM.ipynb) and analysis of the results alongside other RL literature is available in a [pdf report](https://github.com/bragancas/texttest/blob/master/DLO.pdf).

## References

1. [Andrew G. Barto Richard S. Sutton. Reinforcement Learning. The MIT Press, Nov. 13, 2018. 552 pp. isbn: 0262039249](https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf)
2. [Greg Brockman et al. Openai gym"](https://arxiv.org/abs/1606.01540)
3. [Volodymyr Mnih et al. Playing Atari with Deep Reinforcement Learning. 2013](https://arxiv.org/abs/1312.5602)

## [Reinforcement Learning overview](https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf)

Reinforcement learning uses a computational approach to automate learning and decision making to achieve goals. It uses a framework where an *agent*(the decision maker) achieves a goal by learning from its interaction with its *environment*(comprises everything outside the agent). This interaction is represented in terms of states, actions and rewards over a sequence of discrete time steps.

At each time step t, the agent receives some representation of the environment's state, S<sub>t</sub> ∈ S, where S is the set of possible states, and on that basis selects an *action*, A<sub>t</sub> ∈ A(S<sub>t</sub>), where A(S<sub>t</sub>) is the set of actions available in state S<sub>t</sub>. One time step later, in part as a consequence of its action, the agent receives a numerical *reward*, R<sub>t+1</sub> ∈ R⊂ℝ, and finds itself in a new state, S<sub>t+1</sub>. The actions are the choices made by the agent, the states are the basis for making the choices, and the rewards are the basis for evaluating the choices. The agent's sole objective is to maximize the total reward it receives over the long run. The reward signal thus defines what are the good and bad events for the agent. The reward sent to the agent at any time depends on the agent's current action and the current state of the agent's environment. The agent cannot alter the process that does this. The only way the agent can influence the reward signal is through its actions, which can have a direct effect on reward, or an indirect effect through changing the environment's state.

Two essential components that help an RL agent in automation towards achieving a goal are policy and value functions. A *policy* is a stochastic rule by which the agent selects actions as a function of states. It defines the agent's way of behaving at a given time. The reward signal is the primary basis for altering the policy. If an action selected by the policy is followed by low reward, then the policy may be altered to select some other action in that situation in the future. Policy is denoted by **π**<sub>t</sub> where **π**<sub>t</sub>(a|s) is the probability that an action 'a' is taken given the agent is in state 's'.

Whereas the reward signal indicates what is good in an immediate sense, a *value function* specifies what is good in the long run. Roughly speaking, the value of a state is the total amount of reward an agent can expect to accumulate over the future, starting from that state. Considering a certain state, rewards determine the immediate, intrinsic desirability of that state, while values indicate the long-term desirability of that state by taking into account the states that are likely to follow, and the rewards they offer. We seek actions that bring about states of highest value(current + future rewards), not highest reward(current reward), because these actions obtain the greatest amount of reward for us over the long run.

## Mountain Car problem

The agent in this environment is the car and the goal is to reach the flag. There are two variants of this environment. One with discrete actions available to the agent as push left, push right, no push and a continuous action variant where the agent can accelerate or decelerate within predefined speed limits. The challenge in this environment is that the agent cannot reach the flag just by pushing right, as it cannot build enough momentum to climb the hill this way. However, it can build enough momentum by first ascending the hill on the left, and then pushing right from there. This is an example of a control task in which an agent needs to initially move in the opposite direction of a target position. Such tasks tend to be challenging without a human designer.

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<img height="200" width="300" align="center" src=https://github.com/bragancas/texttest/blob/master/plots/Mountain_car_gif.gif> 

## Algorithm 1: Q-learning 

### Q-Learning Pseudocode & Algorithm overview

Q-learning uses temporal-difference learning in environments that can be represented as a Markov Decision Process. The learning is an iterative process defined below:

<ins>***Q-learning for estimating optimal policy*** ( **π**</ins><sub>*</sub>)\
Initialize Q(s,a), ∀ s ∈ S, a ∈ A(s), arbitrarily, and Q(terminal-state, **·**) = 0\
Repeat (for each episode):\
&emsp;Initialize S\
&emsp;Repeat (for each step of episode):\
&emsp;&emsp;&emsp;Choose A from S using policy derived from Q (e.g., **ϵ**-greedy)\
&emsp;&emsp;&emsp;Take action A, observe R, S<sub>t+1</sub>\
&emsp;&emsp;&emsp;Q<sup>new</sup>(S<sub>t</sub>,A<sub>t</sub>) ⟵ Q<sup>old</sup>(S<sub>t</sub>,A<sub>t</sub>) + **α**( R<sub>t+1</sub> + **γ** max<sub>A</sub>Q(S<sub>t+1</sub>, A) - Q<sup>old</sup>(S<sub>t</sub>,A<sub>t</sub>))\
&emsp;&emsp;&emsp;S ⟵ S<sub>t+1</sub>\
&emsp;until S is terminal
	
Where,

<ins>Q-value Q(S<sub>t</sub>,A<sub>t</sub>)</ins>: This variable is the *state-action value function* under a given policy, and represents the expected discounted cumulative reward of taking an *action* in a given state. This is initialised to 0 for all state-action pairs, and updated iteratively as specified in the algorithm above.

<ins>Learning rate</ins> (**α**): This parameter determines the degree by which the algorithm updates exisiting Q-values, i.e. the pace of it's learning, and ranges between 0 and 1. A value of 0 means the agent doesn't uptake the reward { immediate[R<sub>t+1</sub>] + discounted future[**γ** max<sub>A</sub>Q(S<sub>t+1</sub>, A)] } it receives and therefore learns nothing at each iteration, and a value of 1 means the agent completely discards previously learned Q-values{ Q<sup>old</sup>(S<sub>t</sub>,A<sub>t</sub>) · (1-**α**) } in favour of new values { R<sub>t+1</sub> + **γ** max<sub>A</sub>Q(S<sub>t+1</sub>, A) · **α** }.

<ins>R<sub>t+1</sub></ins>: The reward received after taking action A<sub>t</sub> when at state S<sub>t</sub>

<ins>max<sub>A</sub>Q(S<sub>t+1</sub>, A)</ins>: This variable represents the maximum expected future cumulative reward from each of the possible state-action pairs. Factoring this value into the algorithm influences the agent to learn which state it must arrive at over others (and therefore which action it must undertake) to achieve a higher reward.

<ins>Discount rate</ins> (**γ**): This parameter quantifes the importance that the agent gives to future rewards when taking an action. If **γ** = 0, the agent only considers immediate rewards(R<sub>t+1</sub>). As **γ** approaches 1, the agent considers future rewards more and more strongly in addition to its immediate reward (the agent is farsighted in its learning and consequently its behaviour in maximising its rewards). As **γ** is decreased, future rewards are discounted more, thus reducing the influence they have over the agent when choosing an action (agent is nearsighted in its capability to maximise its rewards).

### Environment 

In the mountain car environment, any given state at time t (S<sub>t</sub>) is represented by a pair of values, position (x) and velocity (v). Both of these values are from closed, bounded sets. Formally: 

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;S<sub>t</sub>=(x, v), where x ∈ [-1.2, 0.6] and v ∈ [-0.07, 0.07]

The goal state is reached when x=0.5. Both elements representing a state are continuous variables. However, Q-learning algorithm needs the environment to be represented as a Markov Decision Process and therefore requires discretising the environment variables of velocity and position the agent can be in at any time. The experiments that follow are carried out using 25 discrete values/bins (chosen arbitrarily) for both these variables yielding a total of 625 states. So for example, the position between -1.2 and -1.125 is treated as one discrete state. For bins, a number too small would over simplify the environment and result in very limited exploration by the agent, whilst a number too large would better resemble the true state space resulting in more effective exploration however at the expense of lengthier computational times.

### Reward

The reward is a key component in the agent's learning process as we want the agent to maximise it's accumulated reward during the learning process. The default reward for each action the agent takes is -1. An exception is when the action taken leads the agent to reach the target, for which the reward is 0. So, the more steps taken to reach the flag(goal state), the lower the reward accumulated. Thus, maximising accumulated reward in our environment translates to taking less steps. 
Formally, the reward function is defined as:

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;R<sub>t+1</sub> = -1, except when the agent reaches the goal state.

### Policy

In any environment it is generally desirable for the agent to explore the various states and the rewards it can achieve across the state-action space so as to incorporate better into its learning which state-action allows earning higher rewards and exploiting these after sufficient exploration. A policy which manages the extent of exploration is the **ϵ**-greedy policy. In this policy, **ϵ** is the parameter which needs to be defined and it refers to the probability that an agent explores the environment, i.e. randomly selects an action. Higher values of **ϵ** equates to a higher probability of the agent taking a random action(resulting in exploration) instead of the best identified action(having highest Q-value) as updated in the Q-values. 

Another optional parameter under the **ϵ**-greedy policy is **λ**, the rate of decay. This is the value by which **ϵ** is reduced by after each episode. Using this will invoke a balance between exploration and exploitation. The motivation for this is that it's desirable for the agent to gradually exploit what it has learned, whilst still exploring albeit at a lower rate. An **ϵ**<sub>min</sub> can also be chosen as a threshold, to avoid employing an entirely exploitative strategy.

### Q-learning parameters

| Parameter                                | Value |
| :---:                                    | :---: |
| Learning Rate (**α**)                   | 0.5 |
| Discount Rate (**γ**)                   | 0.8 |
| Policy                                   | **ϵ**-greedy |
| Epsilon (**ϵ**)                       | 0.5 |
| Epsilon Decay (**λ**)                  | 0.002 per episode |
| Minimum Epsilon (**ϵ**<sub>min</sub>) | 0.01 (will be reached after 245  epsiodes) |
| Number of episodes                       | 5,000 |
| Max. number of steps per episode         | 25,000 |


### Illustration of Q-values being updated

The updates to Q-values follows the pseudo-algorithm mentioned previously and is carried out in the manner listed below.

1. Initialise all Q-values as zeros and set the required parameters.
2. Choose an initial action from A=[0,1,2] according to the **ϵ**-greedy policy.
3. Perform the action and receive reward R<sub>0</sub> and enter next state S<sub>1</sub>.
4. Determine the future reward as max<sub>A</sub>Q(S<sub>t+1</sub> and update the Q-value for Q(S<sub>t</sub>,A<sub>t</sub>) using the equation in the pseudo-algorithm.
5. Repeat above steps 2-4 till the goal state is reached or 25,000 steps have been taken.

| Time Step | Variables for each Time Step | Dictionary Key<sup>Python</sup> | &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Dictionary Value &emsp;&emsp;&emsp;&emsp;&emsp;&emsp; {Q(S<sub>t</sub>,A<sub>t</sub>) + **α**( R<sub>t+1</sub> + **γ** max<sub>A</sub>Q(S<sub>t+1</sub>, A) - Q(S<sub>t</sub>,A<sub>t</sub>)}|
| :---: | :---: | :---: | :---: |
| t = 0 | S<sub>0</sub> = (24,24), A<sub>0</sub> = 0, R<sub>1</sub> = -1 &nbsp;&nbsp;S<sub>1</sub> = (10,12), max<sub>A</sub>(S<sub>1</sub>,A) = 0 | Q{(24,24), 0} | 0 + 0.5(-1 + 0.8 x 0 - 0) = -0.5 |
| t = 1 | S<sub>1</sub> = (10,12), A<sub>1</sub> = 0, R<sub>2</sub> = -1 &nbsp;&nbsp; S<sub>2</sub> = (10,12), max<sub>A</sub>(S<sub>2</sub>,A) = 0 | Q{(10,12), 0} | 0 + 0.5(-1 + 0.8 x 0 - 0) = -0.5 |
| t = 2 | S<sub>2</sub> = (10,12), A<sub>2</sub> = 1, R<sub>3</sub> = -1 &nbsp;&nbsp;S<sub>3</sub> = (10,12), max<sub>A</sub>(S<sub>3</sub>,A) = 1 | Q{(10,12), 1} | 0 + 0.5(-1 + 0.8 x 0 - 0) = -0.5 |
| t = 3 | S<sub>3</sub> = (10,12), A<sub>3</sub> = 2, R<sub>4</sub> = -1 &nbsp;&nbsp;S<sub>4</sub> = (10,12), max<sub>A</sub>(S<sub>4</sub>,A) = 2 | Q{(10,12), 2} | 0 + 0.5(-1 + 0.8 x 0 - 0) = -0.5 |
| t = 4 | S<sub>4</sub> = (10,12), A<sub>4</sub> = 1, R<sub>5</sub> = -1 &nbsp;&nbsp;S<sub>5</sub> = (10,12), max<sub>A</sub>(S<sub>5</sub>,A) = 0 | Q{(10,12), 1} | -0.5 + 0.5(-1 + 0.8 x -0.5 - (-0.5)) = -0.95 |

The agent begins at state (24,24), and takes action 0. This takes the agent to state (10,12), from which the agent takes actions 0, 1, 2 and then 1 again, all which result in the agent remaining in state (10,12). However due to the key-value structure, separate Q-Values are stored for each state-action pair.

### Experimental Results for Q-Learning
<ins>(**Note**: See attached [pdf report](https://github.com/bragancas/texttest/blob/master/DLO.pdf) for analysis)</ins>

**1)** Varying Learning Rate to measure effect on Q-Learning Rewards over Episodes (1)

<img height="400" width="6900" align="centre" src=https://github.com/bragancas/texttest/blob/master/plots/Varying_Learning_rate(1).png>

**2)** Varying Learning Rate to measure effect on Q-Learning Rewards over Episodes (2)

<img height="400" width="900" align="centre" src=https://github.com/bragancas/texttest/blob/master/plots/Varying_Learning_Rate_(2).png>

**3)** Varying Discount Rate to measure effect on Q-Learning Rewards over Episodes (1)

<img height="400" width="900" align="centre" src=https://github.com/bragancas/texttest/blob/master/plots/Varying_Discount_Rate_(1).png>

**4)** Varying Discount Rate to measure effect on Q-Learning Rewards over Episodes (2)

<img height="400" width="900" align="centre" src=https://github.com/bragancas/texttest/blob/master/plots/Varying_Discount_Rate_(2).png>

**5)** Varying Epsilon to measure effect on Q-Learning Rewards over Episodes (1)

<img height="400" width="900" align="centre" src=https://github.com/bragancas/texttest/blob/master/plots/Varying_Epsilon_(1).png>

**6)** Varying Epsilon to measure effect on Q-Learning Rewards over Episodes (2)

<img height="400" width="900" align="centre" src=https://github.com/bragancas/texttest/blob/master/plots/Varying_Epsilon_(2).png>

**7)** Varying Policy to measure effect on Q-Learning Rewards over Episodes (1)

<img height="400" width="900" align="centre" src=https://github.com/bragancas/texttest/blob/master/plots/Varying_Policy_Used_(1).png>

**8)** Varying Policy to measure effect on Q-Learning Rewards over Episodes (2)

<img height="400" width="900" align="centre" src=https://github.com/bragancas/texttest/blob/master/plots/Varying_Policy_Used_(2).png>


## Algorithm 2: Deep Q-Network(DQN) with Experience Replays

The nature of the Q-learning algorithm required the state and action space of the mountain car environment to be discrete. This restricted the number of bins used to represent this state-action space, as finely discretising the state-action space would result in storing and utilising a large number of 'Q-value Dictionary keys' to learn from. With Deep Reinforcement Learning methods such as DQN we approximate these Q-values as a non-linear function using a neural network. By using a NN, we can provide continuous values representing the state space as input to the network and overcome the restrictions of working with a "manageable" state space and are thus capabile of handling large state spaces. 

### Algorithm overview

DQN employs the technique of *experience replays* where once the networks(train and target networks, 2 nets for more [stable training](https://www.nature.com/articles/nature14236)) are initialised with random weights, the initial episode is begun and the agent's *experiences* for several time steps, e<sub>t</sub> = (s<sub>t</sub>, a<sub>t</sub>, r<sub>t+1</sub>, s<sub>t+1</sub>) are retained. At timestep *t*, retain information about the initial state(s<subt></subt>) the agent was in, the action(a<sub>t</sub>) taken as determined by the randomly initialised training network weights, the immediate reward(r<sub>t+1</sub>) received when undertaking that action, and the next state(s<sub>t+1</sub>) after taking the action. These experiences are stored in the ***'replay memory'*** with the total number of experiences to be retained in this memory to be specifed. Here, the actions taken(random or from network o/p prediction) are in line with **ϵ**-greedy policy with **ϵ** = 0.5 and **λ** = 0.02 per episode.

After sufficient experiences have been gathered, randomly chosen samples(commonly referred as mini-batch) are used from the replay memory for training the network. These samples are referred as the ***'memory buffer'***. Random samples are used for training as consecutive samples would cause the training data to be highly correlated and is undesirable as it can introduce variance when updating the network's weights between each successive training iteration. The network weights are updated using an Adam optimiser with loss as the mean square of temporal-difference error(defined below). 

During a training iteration, for each experience in the memory buffer, we pass s<sub>t</sub> into the training network and s<sub>t+1</sub> into the target network. Their output predictions correspond to the estimated Q-values(Q(s<sub>t</sub>,a<sub>t</sub>) for training net and Q'(s<sub>t+1</sub>,A) for target net) for each available action(0,1,2) the agent can take. For each of the undertaken actions in the experiences within the memory buffer, we optimise the Q-value predictions of the training network using the *temporal-difference error* 

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;error = { r<sub>t+1</sub>, for terminal state s<sub>t+1</sub>\
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;&emsp;&emsp;&emsp;{ r<sub>t+1</sub> + **γ**max<sub>A</sub>Q'(s<sub>t+1</sub>,A) - Q(s<sub>t</sub>,a<sub>t</sub>), otherwise

At the end of an episode (target state or until max iterations) the weights of the target network is updated with that of the training network and the training process is continued till the max number of episodes are completed. With this learning methodology, initially the agents actions are poor but over time as the agent gathers more experiences the amount of quality labels used to train the networks improve and so does the network training/learning over time.

Instances where the sampled experiences could cause a large change in training network weights with gradient updates could lead to oscillation of the actions predicted by the network or training/learning algorithm attaining a suboptimal minimum. Using a second target network whose weights are updated only at the end of the episode stabilises training when encountering such possible instances as the Q-updates (training network weight updates) are dependent on the older target network weights which remain the same over the episode. 

### DQN Pseudocode
The DQN algorithm is presented with the pseudo-code below:

<ins>***Deep Q-learning with experience replay***</ins>\

Initialize training network Q\
Initialize target network Q'\
Initialize replay memory D\
Repeat (for each episode):\
&emsp;Initialize S\
&emsp;Repeat (for each step/iteration of episode):\
&emsp;&emsp;&emsp;Choose Action a<sub>t</sub> randomly with probability **ϵ**, else a<sub>t</sub> = max<sub>A</sub>Q(s<sub>t</sub>,a<sub>t</sub>)\
&emsp;&emsp;&emsp;Take action a<sub>t</sub>, observe r<sub>t+1</sub> and s<sub>t+1</sub>\
&emsp;&emsp;&emsp;Store experience (s<subt></subt>, a<sub>t</sub>, r<sub>t+1</sub>, s<sub>t+1</sub>) in memory D\
&emsp;&emsp;&emsp;If retained sufficient experiences in D\
&emsp;&emsp;&emsp;&emsp;&emsp;Sample a random minibatch of N experiences as memory buffer from D\
&emsp;&emsp;&emsp;&emsp;&emsp;For every experience e<sub>i</sub> = (s<subt></subt>, a<sub>t</sub>, r<sub>t+1</sub>, s<sub>t+1</sub>) in minibatch\
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;if terminal state then set y<sub>i</sub> = r<sub>t+1</sub>, else y<sub>i</sub> = r<sub>t+1</sub> + **γ**max<sub>A</sub>Q'(s<sub>t+1</sub>,A)\
&emsp;&emsp;&emsp;&emsp;&emsp;Determine loss L = 1/N·(y<sub>i</sub> - Q(s<sub>t</sub>,a<sub>t</sub>))<sup>2</sup>\
&emsp;&emsp;&emsp;&emsp;&emsp;Minimise Loss using Adam optimiser and update training network weights\
&emsp;&emsp;&emsp;Update current state s<sub>t</sub> = s<sub>t+1</sub>\
&emsp;&emsp;&emsp;Until max number of iterations or terminal state attained\
&emsp;Update target network weights with training network\
&emsp;until max episodes completed



### Deep Q-Network parameters

| Parameter                                | Value |
| :---:                                    | :---: |
| Discount Rate (**γ**)                   | 0.99 |
| Epsilon (**ϵ**)                       | 0.5 |
| Epsilon Decay (**λ**)                  | 0.002 per episode |
| Minimum Epsilon (**ϵ**<sub>min</sub>) | 0.01 (will be reached after 245  epsiodes) |
| ------------------------------------- | ------------------------------------------------------- |
| Network Architecture | {Input-2,24[Dense,ReLU]},{24,48[Dense,ReLU]},{48,Output-3[Dense,Linear]} |
| Learning rate | 0.001 |
| Replay memory size | 20,000 |
| Memory buffer size | 120 |
| ------------------------------------- | ------------------------------------------------------- |
| Number of episodes                       | 5,000 |
| Max. number of steps per episode / Training iterations         | 1,000 |


### Experimental Results for Deep Q-Network
<ins>(**Note**: See attached [pdf report](https://github.com/bragancas/texttest/blob/master/DLO.pdf) for analysis)</ins>

**1)** Varying Epsilon to measure effect on Rewards over Episodes

<img height="400" width="900" align="centre" src=https://github.com/bragancas/texttest/blob/master/plots/DQN_varying_Epsilon.png>

**2)** Varying Learning Rate to measure effect on Rewards over Episodes

<img height="400" width="900" align="centre" src=https://github.com/bragancas/texttest/blob/master/plots/DQN_Varying_Learning_Rate.png>


## Algorithm 3: Advantage Actor-Critic (A2C)

Just as with DQN, A2C makes use of neural networks as non linear function approximators to overcome the limitations of traditional Q-learning. However, in A2C we approximate not just a value function (as was in DQN) but also the policy as a function which maps states to actions. Doing this allows us to output actions as continuous values which can be optimised. This is especially useful for agents in real world environments where taking discrete actions would reduce manoeuvrability for example. The environment used for this algorithm is the [continuous Mountain car](https://gym.openai.com/envs/MountainCarContinuous-v0/) variant. Here the agent's continuous action space can take values between -1 and 1 which dictates the amount of force used to move left or right respectively. The agent also receives a reward 100 upon reaching the goal state. The reward is now defined as:

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;R<sub>t+1</sub> = { -(applied force)<sup>2</sup> * 0.1, applied force ∈ [-1.0, 1.0]\
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;{ 100, upon reaching goal state 

### Algorithm overview
In A2C we have two networks, the actor network and the critic network. The role of the actor network is to model an optimal policy with the critic network influencing the learning of the actor network to reach this optimal policy. This can be understood through the objective function of the actor network which follows the policy gradient theorem: 

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;∇<sub>θ</sub>J(θ) = E [∇<sub>θ</sub> log π<sub>θ</sub>(a|s) · Q<sup>π<sub>θ</sub></sup>(s,a)]

Here,

∇<sub>θ</sub> is the gradient w.r.t to actor network parameters/weights θ\
J(θ) is the objective we wish to maximise and is a function of the parameters/weights θ\
π<sub>θ</sub>(a|s) is the policy function parametrised by the network weights θ and is the probability of taking an action 'a' when in a certain state 's'\
Q<sup>π<sub>θ</sub></sup>(s,a) is the state-action value function of taking the action 'a' in state 's' when following policy π\
E is the expectation over states and actions of the enclosed\

The above equation implies that for a sample of the expectation, those actions given by the policy 'π<sub>θ</sub>' which result in a higher Q-value Q<sup>π<sub>θ</sub></sup>(s,a) influence the gradient so as to update actor's weights more in the direction of the policy that generated the high Q-value acheiving action. The critic network approximates the state-action value function (Q<sub>U</sub>(s,a)) ≈ Q<sup>π<sub>θ</sub></sup>(s,a)).

While approximating Q-values and utilising them to improve the policy works succesfully, an improvement can be made to the learning process to speed up learning. Utilising Q-values for actor improvement introduces high variance in the learning process as the Q-values can vary highly with each sample as the value consists of cumulative rewards across all future states in a trajectory, which could produce subsequent high positive & negative or high positive followed by even higher positive returns. This can slow down convergence of training as gradients would be updated drastically at each subsequent iteration. To overcome this a baseline is subtracted from the Q-value and a suitable baseline is the 'State Value function'(as it doesnt depend on action and only state which results in no change to the expectation[refer sutton and barto REINFORCE with baseline]). The state value function or V-value is the expected discounted cummulative reward for a given state or the weighted sum of Q-values, weighted by the policy(the probability of actions in a state). While Q-values pertain to cumulative rewards taking a specific *action* when in a given state, V-values consider the cumulative reward the agent can achieve by virtue of just being in that state. Subtracting the V-value of a state from the Q-value provides a measure of how important a particular *action* is for the *current state* over the weighted sum of all actions for the current and all future states. With this we can eliminate variance introduced while asserting meaningful influence over the actors gradient. The resulting value is known as the advantage function A<sup>π<sub>θ</sub></sup>(s,a) =  Q<sup>π<sub>θ</sub></sup>(s,a) - V<sup>π<sub>θ</sub></sup>(s) which can be approximated as TD-error δ<sup>π<sub>θ</sub></sup>. The objective function can be re-written as

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;∇<sub>θ</sub>J(θ) = E [∇<sub>θ</sub> log π<sub>θ</sub>(a|s) · δ<sub>U</sub>]

where,   δ<sub>U</sub> = R<sub>t+1</sub> + γ·V<sub>U</sub>(s<sub>t+1</sub>) - V<sub>U</sub>(s<sub>t</sub>)

The input to the actor network is the continuous state comprised of position and velocity of the car. For the input state the actor network outputs two values as mean and standard deviation of a Gaussian distribution. This distribution is the stochastic policy which is sampled to determine the action to take in the environment yielding response as reward and next state information. Using this response to calculate the TD error specified above, we minimise the critic's error against the target action value function (R<sub>t+1</sub> + γV<sub>U</sub>(s<sub>t+1</sub>)). In doing so the critic is able to learn the expected sum of discounted rewards for a given state and influences the actor's learning so that the actor generates actions that maximise the reward and consequently its objective function.


### A2C Pseudocode
The A2C algorithm is carried out according to the pseudo-code below:

<ins>***Advantage Actor Critic***</ins>\
Initialize actor network with weights θ\
Initialize critic network with weights U\
Repeat (for each episode):\
&emsp;Initialize S\
&emsp;Repeat (for each step/iteration of episode):\
&emsp;&emsp;&emsp;Sample Action a<sub>t</sub> ~ π(a| µ(s<sub>t</sub>), σ(s<sub>t</sub>)) = N(a| µ(s<sub>t</sub>), σ(s<sub>t</sub>))\
&emsp;&emsp;&emsp;Take action a<sub>t</sub>, observe R<sub>t+1</sub> and s<sub>t+1</sub>\
&emsp;&emsp;&emsp;Determine TD target y<sub>t</sub> = R<sub>t+1</sub> + γV<sub>U</sub>(s<sub>t+1</sub>)\
&emsp;&emsp;&emsp;Update critic weights by minimising δ<sub>t</sub> = (y<sub>t</sub> - V<sub>U</sub>(s<sub>t</sub>))<sup>2</sup>\
&emsp;&emsp;&emsp;Update actor weights by minimising  -log N(a| µ(s<sub>t</sub>), σ(s<sub>t</sub>)) · δ<sub>t</sub> \
&emsp;&emsp;&emsp;Update current state s<sub>t</sub> = s<sub>t+1</sub>\
&emsp;&emsp;&emsp;Until max number of iterations or terminal state attained\
&emsp;until max episodes completed

### A2C parameters

| Parameter                                | Value |
| :---:                                    | :---: |
| Discount Rate (**γ**)                   | 0.99 |
| ------------------------------------- | ------------------------------------------------------- |
| Actor Network Architecture | {Input-1,40[Dense,eLU]},{40,40[Dense,eLU]},{40,Output-2[Dense,Linear]} |
| Critic Architecture | {Input-1,400[Dense,eLU]},{400,400[Dense,eLU]},{400,Output-1[Dense,Linear]} |
| Actor Learning rate | 0.0001, 0.00005, 0.00001 |
| Critic Learning rate | 0.005, 0.0008, 0.00056 |
| ------------------------------------- | ------------------------------------------------------- |
| Number of episodes                       | 1,000 |
| Max. number of steps per episode / Training iterations         | 1,000 |

### Experimental Results for A2C
<ins>(**Note**: See attached [pdf report](https://github.com/bragancas/texttest/blob/master/DLO.pdf) for analysis)</ins>

**1)** Rewards Vs Episodes (Successful Run)

<img height="400" width="900" align="centre" src=https://github.com/bragancas/texttest/blob/master/plots/A2C_rewards_successful.png>

**2)** Steps Vs Episodes (Successful Run)

<img height="400" width="900" align="centre" src=https://github.com/bragancas/texttest/blob/master/plots/A2C_num_steps_taken_succesful.png>

**3)** Rewards Vs Episodes (Unsuccessful Run)

<img height="400" width="900" align="centre" src=https://github.com/bragancas/texttest/blob/master/plots/A2C_rewards_unsuccessful.png>

**4)** Steps Vs Episodes (Unsuccessful Run)

<img height="350" width="900" align="centre" src="https://github.com/bragancas/texttest/blob/master/plots/A2C steps taken unsucceful.png">
