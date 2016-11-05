import os
import tasks
import models
import agents
from analysis import Analysis

###########
# Parameter specification

train_iter = 1*10**4 # number of training iterations
test_iter = 1*10**3 # number of test iterations

###########
# Environment specification

task = tasks.ProbabilisticCategorization()

###########
# Model specification

model = models.RNN_Elman(task.ninput, 5, task.noutput)

##########
# Agent specification

#agent = agents.AACAgent(model)
agent = agents.SupervisedAgent(model)

###########
# Train agent on an environment

result = agent.learn(task, train_iter)

###########
# Analyse training behaviour

fname = 'figures/learn_probabilistic_categorization'

analysis = Analysis(fname, task, agent)

analysis.cumulative_reward(result['reward'])

###########
# Run agent on an environment

result = agent.run(task, test_iter)

###########
# Analyse testing behaviour

fname = 'figures/run_probabilistic_categorization'

analysis = Analysis(fname, task, agent)

analysis.cumulative_reward(result['reward'])

[U,W] = model.policy_weights()
analysis.weight_matrix(W)

analysis.functional_connectivity(result['hidden_pi'])

#analysis.spike_rate(name, result['hidden'], result['terminal'])