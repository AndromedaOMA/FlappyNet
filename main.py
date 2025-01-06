# from scripts import Agent
from scripts import Agent2

"""DQN"""
# agent = Agent("./best_models/DQN/trained_q_function_14.000.pth")

"""Dueling-DQN"""
# agent = Agent("./best_models/DuelingDQN/trained_q_function_10.100.pth")  # it works fine

"""CNN-DQN (deprecated)"""
# agent = Agent2("./best_models_CNN/CNN/trained_q_function_???.pth") # still working

"""Dueling-CNN-DQN"""
agent = Agent2("./best_models_CNN/DuelingCNN/last_state/trained_q_function_final")
# agent = Agent2("./best_models_CNN/DuelingCNN/last_state/trained_q_function")

agent.run(is_training=False, render=True)
# agent.run(is_training=True, render=False)

