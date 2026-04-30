#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing all the python libraries
import numpy as np
import time
import gym
from gym import envs
import random
import datetime
from IPython.display import clear_output
from time import sleep


# The goal of Taxi game environment to be sure for the passenger to pick up and drop - off the passenger in the fastest way possible.
# - Reward: Penalty linked to that particular action
# - Done: If Yes, it means a passenger has picked up and dropped off, this is also called episodes, it means the episodes  is over otherwise it is not.
# - Observation: Observation of the environment

# In[2]:


#Creating the environment
env = gym.make("Taxi-v3")


# Render: Renders the environment,it helps in visualizing the envirnoment.
# 
# Reset: Reset the environment and returns to the intial position.
# 
# Possible Actions -
# 
# - 0: move south
# - 1: move north
# - 2: move east
# - 3: move west
# - 4: pickup passenger
# - 5: dropoff passenger
# 
# Location is referred as R,G,Y,B.
# 
# Taxi environment has 5×5×5×4 = 500 Total states.
# - 5 = rows
# - 5 = Column
# - 4+1= five passenger location
# - 4 = Destination

# In[3]:


#Show it
env.render() 
env.reset()


# # Define Discrete States and Action

# In[4]:


# Print Number of states 
(env.observation_space.n)


# In[5]:


# Print Number of actions
env.action_space.n  


# In[6]:


print("Action {}".format(env.action_space))
print("State {}".format(env.observation_space))
env.render()


# - Pipe("|") represents the wall which taxi cannot cross
# - R,G,Y,B are the pick up and drop - off destination for the passenger.
# - Yellow square represnt the Taxi
# - Blue Letter and Purple letter represnts the current passenger pick - up and drop - off location
# - The filled sqaure represnts the Taxi.

# # The Reward matrix

# In[7]:


state = env.encode(4, 3, 2, 1) # (taxi row, taxi column, passenger index, destination index)
print("State:", state)
env.s = state
env.render()


# Above we can see that, taxi position is (4,3) as it has mentioned taxi row and column, and that passenger index is 2 and drop - off destination is 1.We can understand that the taxi state is 469.
# The rewarding system is the main idea in the reinforcement learning and agents is reward well when it works correctly or for the incorrect behaviour it will be rewarded newgatively.
# When the Taxi environment is created, the embedded reward table is created which is Called 'P'. The logic is below:
# - When the taxi correctly pick up and drop - off of the passenger, it is rewarded with + 20 points.
# - If the taxi take up a wrong pick up and drop - off of the passenger, it is  rewarded with -10 points.
# - For each step which will not count state of the taxi it is reward with -1 points. 

# In[8]:


env.P[469]


# In the above P table matrix: action, probability, nextstate, reward, done.
# - Action is 0 to 5 (South, North, East, West, Pick up and drop - off)
# - Probability is always 1 
# - Nextstate, if the state occurs and action is done 
# - Reward, penalty links to that aciton
#  - All the actions have -1 rewards, pick up and drop-off has - 10 reward in this particular state.
# - Done, Its means the passenger is successfully dropped - off and the episode is over.

# In[9]:


env.s = 469  # set environment to demonstration state

epochs = 0
penalties, reward = 0, 0

frames = [] # for animation

done = False

while not done:
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)

    if reward == -10:
        penalties += 1
    
    # Put each rendered frame into dict for animation
    frames.append({
        'frame': env.render(mode='ansi'),
        'state': state,
        'action': action,
        'reward': reward
        }
    )

    epochs += 1
    
print("Timesteps taken: {}".format(epochs))
print("Penalties incurred: {}".format(penalties))


# In[10]:


from IPython.display import clear_output
from time import sleep

def print_frames(frames):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame['frame'])
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(.1)
        
print_frames(frames)


# The taxi takes thousands of steps and makes lots of wrong drop offs to carry just one passenger to the right destination.
# 
# This is because we aren't learning from past experience. We can run this over and over, and it will never optimize. The agent doesn't have memory of which action was best for each state, which is exactly that Reinforcement Learning will do for us.

# Lets enter the reinforcement Learning 

# We will train the agent on reinforcement learning which is Q - learning algorithm to give some memory.

# # Implementing the Q- learning Algorithm
# 
# Q-learning lets the agent use the environment's rewards to learn, over time, the best action to take in a given state.
# 
# In our Taxi environment, we have the reward table, P, that the agent will learn from. It does thing by looking receiving a reward for taking an action in the current state, then updating a Q-value to remember if that action was beneficial.
# 
# The values store in the Q-table are called a Q-values, and they map to a (state, action) combination.
# 
# A Q-value for a particular state-action combination is representative of the "quality" of an action taken from that state. Better Q-values imply better chances of getting greater rewards.
# 
# For example, if the taxi is faced with a state that includes a passenger at its current location, it is highly likely that the Q-value for pickup is higher when compared to other actions, like dropoff or north.

# # Q- Table
# The Q-table is a matrix where we have a row for every state (500) and a column for every action (6). It's first initialized to 0, and then values are updated after training. Note that the Q-table has the same dimensions as the reward table, but it has a completely different purpose.

# In[11]:


#Intializing the Q- learning algorithm to train the agents.
#Q-table to a 500×6 matrix of zeros:
q_table = np.zeros([env.observation_space.n, env.action_space.n])


# In[12]:


q_table.shape


# We can now create the training algorithm that will update this Q-table as the agent explores the environment over thousands of episodes.
# 
# In the first part of while not done, we decide whether to pick a random action or to exploit the already computed Q-values. This is done simply by using the epsilon value and comparing it to the random.uniform(0, 1) function, which returns an arbitrary number between 0 and 1.
# 
# I execute the chosen action in the environment to obtain the next_state and the reward from performing the action. After that, we calculate the maximum Q-value for the actions corresponding to the next_state, and with that, we can easily update our Q-value to the new_q_value:

# In[13]:


# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1

# plotting metrics
all_epochs = []
all_penalties = []

for i in range(1, 100001):
    state = env.reset()

    epochs, penalties, reward, = 0, 0, 0
    done = False
    
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explore action space
        else:
            action = np.argmax(q_table[state]) # Exploit learned values

        next_state, reward, done, info = env.step(action) 
        
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        if reward == -10:
            penalties += 1

        state = next_state
        epochs += 1
        
    if i % 100 == 0:
        clear_output(wait=True)
        print(f"Episode: {i}")

print("Training finished.\n")


# Now that the Q-table has been established over 100,000 episodes, let's see what the Q-values are at our demonstration state:

# In[14]:


q_table[469]


# Let's evaluate the performance of our agent. We don't need to explore actions any further, so now the next action is always selected using the best Q-value:

# In[15]:


total_epochs, total_penalties = 0, 0
episodes = 100

for _ in range(episodes):
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0
    
    done = False
    
    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)

        if reward == -10:
            penalties += 1

        epochs += 1

    total_penalties += penalties
    total_epochs += epochs

print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")


# In[16]:


state = env.reset()
done = None

while done != True:
    #take the action with the highest Q Value
    action = np.argmax(q_table[state])
    state, reward, done, info = env.step(action)
    env.render()

