# Deep Reinforcement Learning with Tensorflow 2.x

## Dependencies
- Tensorflow 2.x
- Tensorflow Probability
- Gymnasium

## Chapter 02: Markov Decision Process
* Value Iteration algorithm: 

    - `./chap02/value_iteration.py`

* Policy Iteration Algorithm:

    - `./chap02/policy_iteration_1.py`
    - `./chap02/policy_iteration_2.py`
    - `./chap02/policy_iteration_3.py`

## Chapter 03: Monte-Carlo Methods

* Plotting function
    - `./chap03/plot_blackjack.py`

* Monte-Carlo Prediction Algorithm
    - `./chap03/mc_prediction.py`

* Monte-Carlo Control Algorithm
    - `./chap03/mc_control.py`
    - `./chap03/mc_control_2.py`

## Chapter 04: Temporal Difference Learning
* Q Learning Algorithm 
    - `./chap04/qlearn.py`

* SARSA Algorithm
    -  `./chap04/sarsa.py`


## Chapter 05: Deep Q Networks

* Dependencies
    - `pip install gymnasium[atari], gymnasium[accept-rom-license], gymnasium[classic-control]`

* DQN/DDQN/PER
    - `./chap05/dqn.py`
    
* Replay Buffers
    - `./chap05/buffer.py`

* Gym Wrapper
    - `./chap05/wrappers.py`

* Problems
    - `./chap05/cp_dqn.py` (CartPole Environment)
    - `./chap05/mc_dqn.py` (MountainCar Environment)
    - `./chap05/atari_dqn.py` (Atari Environments)


## Chapter 06: Policy Gradient Methods

* Dependencies:
    - `pip install gymnasium[box2d], swig`

* Monte-Carlo Policy Gradient
    - `./chap06/reinforce.py`

* Deep Deterministic Policy Gradient (DDPG)
    - `./chap06/ddpg.py`

* Proximal Policy Optimization (PPO)
    - `./chap06/ppo.py`

* Utilities
    - `./chap06/utils.py` 

* Problems
    - `./chap06/cp_reinforce.py` (CartPole-v0)
    - `./chap06/lunarlander_reinforce.py` (LunarLander-v2)
    - `./chap06/pendu_ddpg.py` (Pendulum-v1)
    - `./chap06/pendu_ppo.py` (Pendulum-v1)
    - `./chap06/lunarlander_ppo.py` (LunarLander-v2-Continuous)