<h1 align="center">Hi 👋, here we have the FlappyNet project</h1>
<h3 align="center">Developed this project together with Marin Andrei (andier13 on GitHub) in the fifth semester of the faculty.</h3>


## Table Of Content
* [FlappyNet: Developed and trained a Convolutional Neural Network for processing the Flappy Bird game using PyTorch framwork](#flappynet)
* [Installation](#inst)

--------------------------------------------------------------------------------
<h1 id="flappynet" align="left">FlappyNet:</h1>

**NOTE**: This project represents the final project supported and realized within the Neueonal Networks laboratories of the Faculty of Computer Science of the Alexandru Ioan Cuza University in Iași, Romania.

<h3 align="left">Here we have the requirement:</h3>

Implement and train a neural network using the Q learning algorithm to control an agent in the Flappy Bird game.

<h3 align="left">Environment:</h3>

You can use a Flappy Bird environment from [here](https://pypi.org/project/flappy-bird-gymnasium/) or [here](https://github.com/Talendar/flappy-bird-gym) or another environment. Why not create your own? If you use a pre-made environment, make sure you can render the environment and interact with it.

<h3 align="left">Specifications:</h3>

You can train the model directly on images (the model receives the pixels) or you can extract helpful features. Based on the input you are using for the model, the maximum score is capped to:

- 20 points: if you provide the game state directly (this might include positions of the pipes, bird, direction, simple distances)
- 25 points: if you provide preprocessed features (this might include more complex features extracted from the image: e.g. sensors/lidar for the bird)
- 30 points: if you use the image as input, eventually preprocessed, if needed (resizing, grayscale conversion, thresholding, dilation, erosion, background removal, etc.)

It is not necessary to implement the neural network from scratch (you can use PyTorch), but you must implement the Q learning algorithm.

<h3 align="left">How does it work?</h3>

  Implemented several scripts that serve the purpose of training the FlappyBird agent. The scripts are placed in two broad typologies: Deep Q Learning (DQL) and Convolutional Deep Q Learning (CDQN). Both follow the same implementation, the only difference is given by the convolutional aspect of CDQN.
  
  At the same time, for each type of architecture, pre-trained models are available from which we can choose to visualize the FlappyBird agent through the graphical interface, through the "FlappyBird-v0" environment for Gymnasium.
  
  Delving into the implementation idea, the solution uses the concepts of Double DQN (DDQN) and Dueling DQN (Dueling Architecture). These approaches significantly help in training the agent.

  
<h3 align="left">The logic behind the code:</h3>

  The final solution presents the implementation of Neural Networks that are focused on Reinforcement Learning concepts through the Q-Learning algorithm. The first implementation ideas will consider only some classic Neural Networks (without Dueling architecture). So these were the initial implementations of Fully Connected, respectively Convolutional Neural Networks.

  The multilayer neural network consists of three hidden layers of 256 perceptrons each followed by a LayerNorm for data normalization. For the first layer we will use the ReLU activation function, and for the second layer the GeLU activation function and a Dropout for regularization.
  
  The convolutional neural network contains 2 convolutional layers followed by one Pooling layer each. The first convolution layer will contain 16 channels and the second layer will contain 32 channels. The transition from each convolution layer to the pooling layer is processed and filtered by means of a ReLU activation function and BatchNorm2d normalization.

  Through these two Neural Networks, for each one separately, a Target Neural Network identical to the initial one was applied and trained, in parallel, through the Experience Replay stack.

  Finally the original Neural Networks were adapted to the Dueling Architecture

* [Table Of Content](#table-of-content)

---

<h3 id="inst" align="left">Installation:</h3>

1. Clone the current repositoy! Now you have the project avalable on your local server!</br>
 Type command: ```git clone git@github.com:AndromedaOMA/FlappyNet.git```
2. Select, open and run the FlappyNet project through PyCharm IDE or the preferred IDE.
3. Have fun!

* [Table Of Content](#table-of-content)

---

- ⚡ Fun fact: **Through this project I developed better the subtle concepts of Reinforcement Learning and Q-Learning!**
