# Multilayer perceptron.
## (Task 2b)

### Overview

The model is trained and tested with a lot of different values (125) for the following parameters: 
 - number of neurons (values = 20,40,60,80,100)
 - learning rate (values = 0.02, 0.04, 0.06, 0.08, 0.1)
 - training epochs (values = 200, 400, 600, 800, 1000).

### Best results
Best accuracy obtained with the following parameters: 
 - number of neurons = 100
 - learning rate = 0.02
 - training epochs = 800.

**The following diagrams show the accuracy rate of the model with the best found parameters on the training and validation set with respect to the training epochs**

**Green plot is the accuracy on the training set and the orange one is on the validation set.**

First iteration
![Alt text](./1_fold.PNG)

Second iteration
![Alt text](./2_fold.PNG)

Third iteration
![Alt text](./3_fold.PNG)

Fourth iteration
![Alt text](./4_fold.PNG)
