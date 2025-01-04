from scripts import Agent

agent = Agent("./best_models/trained_q_function_14.000.pth")
# agent.run(is_training=False, render=True)
agent.run(is_training=True, render=False)

