# Project

The project is about a paper written by Rico Jonschkowski and Oliver Brock. The goal is to learn state representations based on images and robotics priors to make a network able to produce high level representation of images.
This approach uses deep neural network to achieve the task of learning representation.
The input images are simulate images of Baxter's head camera. The camera is moving from right to left or left to right. The ouput of the neural network should be a representation in one dimension of the head joint. 
![Data example](/Data/pose10_head_pan/Images/frame0010.jpg)<br />
*Example of data from the simulation program*


# Training 

To train the neural network we use the robotics priors from the article "Learning State Representations with Robotic Priors".
We also use siamese networks to apply those priors.
All the data is in this github's folder.
The output of the training should be a value for a given image strongly correlated with the true value of joint.
In training, we achieved 97% of correlation between the real signal of the head and the estimate signal. (the correlation is calculate between signal after normalization of the mean to 0 and std to 1 for the two signals)
