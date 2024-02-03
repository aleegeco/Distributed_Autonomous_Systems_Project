![Minimum Python version >= 3.0](https://badgen.net/badge/python/3.x/blue)
![License](https://badgen.net/badge/license/GPL-3.0/red)


# Distributed Autonomous Systems Course Project
This GitHub page contains solutions to [Project N°1](https://github.com/aleegeco/Distributed_Autonomous_Systems_Project/blob/main/Project_1.pdf) of the [Distributed Autonomous Systems course](https://www.unibo.it/en/teaching/course-unit-catalogue/course-unit/2021/454490).

## Task 1: Distributed Classification via Neural Networks
In this task our goal is to compute the backpropagation of a Neural Network to solve a basic classification problem, but in a Distributed framework where more than a NN concur to 
correctly identify the chosen digit.
Starting from the well known MNIST dataset, we developed from scratch a Neural Network able to solve the classification problem of digits given in input with an accuracy of more than 98%. The NN structure is shown in the figure below: 
![structure_of_NN_DAS](https://github.com/aleegeco/Distributed_Autonomous_Systems_Project/assets/59066644/01a3a312-15a6-4080-91b9-0150a5124a01)
The basic problem has been solved using first a single NN as done in common. Then, we moved forward introducing the distributed framework (by using a communication graph) and solving the task by using
the *distributed gradient tracking* to reach consensus among several agents, where each one should train its own Neural Network.  
The executable *single_neural_network.py* contains the single NN, whereas *distributed_neural_network.py* implements the
distributed gradient tracking simulating N different neural networks trying to reach consensus. 

In-depth results may be seen in the first chapter of [report](https://github.com/aleegeco/Distributed_Autonomous_Systems_Project/blob/main/Distributed_Autonomous_Systems___Report.pdf).
## Task 2: Formation Control by Bearing Based Maneuvering
The second part of the project, deals with a formation control of an arbitrary number of agents. First, we have to define 
the shape that the agents should maintain, then we have to compute appropriate control actions to make it possible.
This kind of problem distinguish between two type of agents: leaders and followers. Leaders motion are always pre-determined, instead followers implement
time-varying control actions to follow the leaders and maintain the formation. 
The model formulation is taken from our reference [1] and a deeper discussion on our decisions is in the second chapter of the [report](https://github.com/aleegeco/Distributed_Autonomous_Systems_Project/blob/main/Distributed_Autonomous_Systems___Report.pdf).

Here a sample video of the result, agents follow the word 'DAS':

https://github.com/aleegeco/Distributed_Autonomous_Systems_Project/assets/59066644/b464b228-ca77-41e7-97f5-b74ebbaec768




### Run Task 2
In order to run this part of the code, it is necessary to have installed *ROS2-foxy* on your computer. 
Type the following command to source ROS2 (consider to add it to ~/.bashrc)
```
source /opt/ros/foxy/setup.bash
```
Then go inside the package folder and build it using
```
cd ~/Formation_Control_task/Formation_Control_task_ROS2 
colcon build
```
If you want to make changes at the code without compiling it again use the following command instead
```
colcon build --symlink-install
```
After compiling it, let's source the install folder
```
source install/setup.bash
```
Finally, if you want to launch the simple formation just use
```
ros2 launch formation_control_task formations.launch.py
```
If you want to launch the word formation instead use
```
ros2 launch formation_control_task formations_letters.launch.py
```
*NB* In both launch files there are some global parameters to change, like integral action, leaders moving or not, which formation you want to perform, ecc..


## Students 
| Student | LinkedIn 
| :-----------: | :--: |
| Alessandro Cecconi | [LinkedIn](https://www.linkedin.com/in/alessandro-cecconi-a5a988182/) |  
| Marco Bugo | [LinkedIn](https://www.linkedin.com/in/marco-bugo/) 
| Roman Sudin | [LinkedIn](https://www.linkedin.com/in/roman-sudin/) 

## References
[1] S. Zhao and D. Zelazo, “Translational and scaling formation maneuver control via a bearing-based approach,” IEEE Transactions on Control of Network Systems, vol. 4, no. 3, pp. 429–438, 2015. [Link of the paper](https://arxiv.org/pdf/1506.05636.pdf)
