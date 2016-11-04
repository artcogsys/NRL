import os
import environments
import modelzoo
import agents
from analysis import Analysis

###########
# Parameter specification

train_iter = 2*10**4 # number of training iterations
test_iter = 1*10**3 # number of test iterations

###########
# Environment specification

env = environments.ProbabilisticCategorization()

###########
# Model specification

model = modelzoo.RNN_Elman(env.ninput, 5, env.noutput)

##########
# Agent specification

agent = agents.Advantage_Actor_Critic(model)

###########
# Train agent on an environment

result = agent.learn(env, train_iter)

###########
# Analyse training behaviour

fname = 'learn_probabilistic_categorization'

analysis = Analysis(fname, env, agent)

analysis.cumulative_reward(result['reward'])

###########
# Run agent on an environment

result = agent.run(env, test_iter)

###########
# Analyse testing behaviour

fname = 'run_probabilistic_categorization'

analysis = Analysis(fname, env, agent)

analysis.cumulative_reward(result['reward'])

[U,W] = model.policy_weights()
analysis.weight_matrix(W)

analysis.functional_connectivity(result['hidden_pi'])

#analysis.spike_rate(name, result['hidden'], result['terminal'])