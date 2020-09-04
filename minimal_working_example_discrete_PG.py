# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Needed for training the network
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.initializers as initializers

# Needed for animation
import matplotlib.pyplot as plt
#epoch_ar = []


    
def get_reward(bandit):
    reward = tf.random.normal \
        ([1], mean=bandit, stddev=1, dtype=tf.dtypes.float32)

    return reward

def plot():
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        bandits = ['1', '2', '3', '4']
        probabilities = [action_probabilities[0,0],\
                    action_probabilities[0,1],\
                    action_probabilities[0,2],\
                    action_probabilities[0,3]]
        ax.bar(bandits,probabilities)

        # Add labels and legend
        plt.xlabel('Episode')
        plt.ylabel('Probability action')
        plt.legend(loc='best')
        
        plt.show()

"""Construct the actor network with mu and sigma as output"""
def construct_actor_network(bandits):
    inputs = layers.Input(shape=(1,)) #input dimension
    hidden1 = layers.Dense(5, activation="relu",kernel_initializer=initializers.he_normal())(inputs)
    hidden2 = layers.Dense(5, activation="relu",kernel_initializer=initializers.he_normal())(hidden1)
    probabilities = layers.Dense(len(bandits), kernel_initializer=initializers.Ones(),activation="softmax")(hidden2)

    actor_network = keras.Model(inputs=inputs, outputs=[probabilities]) 
    
    return actor_network

def cross_entropy_loss(probability_action, state, reward):   
    log_probability = tf.math.log(probability_action + 1e-5)
    loss_actor = - reward * log_probability
    
    return loss_actor


"""Main code"""
# Fixed state
state = tf.constant([[1]],dtype=np.float32)
bandits = np.array([1.0,0.9,0.9,1.0])

# Construct actor network
actor_network = construct_actor_network(bandits)


opt = keras.optimizers.Adam(learning_rate=0.001)


for i in range(10000 + 1):    
    with tf.GradientTape() as tape:  
         # Obtain action probabilities from network
        action_probabilities = actor_network(state)
        
        # Select random action based on probabilities
        action = np.random.choice(len(bandits), p=np.squeeze(action_probabilities))
        
        # Obtain reward from bandit
        reward = get_reward(bandits[action])  

        # Store probability of selected action
        probability_action = action_probabilities[0, action]
        
        # Compute cross-entropy loss
        loss_value = cross_entropy_loss(probability_action, state, reward)

        # Compute gradients
        grads = tape.gradient(loss_value[0], actor_network.trainable_variables)
        
        #Apply gradients to update network weights
        opt.apply_gradients(zip(grads, actor_network.trainable_variables))
        
        
    
    if np.mod(i, 100) == 0:       
        print('\n======episode',i, '======')
        print('probability',float(probability_action))
        print('action',int(action))
        print('reward',float(reward))
        print('loss',float(loss_value))
        
        plot() 