# Logistic-Regression
Logistic Regression

We will implement a logistic regression for multi-class classification: binary-label implementation

1, Prediction
For a given weight w and a new feature vector x, predict the class. Implement in python. Use matrix operation so you can do prediction on a set of test data without loop.
2, Training
Learning the weights by minimizing the cross-entropy loss function.
1)	Derive the gradient formula.
2)	Implement functions to evaluate the gradient and the loss. Remember to use matrix operation.
3)	Implement the steepest descent (using gradient with a stepsize)
•	Hint: start with an initial stepsize (parameter to tune), increase the stepsize by a factor of 1.01 each iteration where the loss goes down, and decrease it by a factor 0.5 if the loss went up. If you are smart you may also undo the last update in that case to make sure the loss decreases every iteration.
4)	To make sure the gradient is calculated correctly, we often validate by comparing with finite-difference approximation. See the following link for a detailed explanation.  
http://ufldl.stanford.edu/tutorial/supervised/DebuggingGradientChecking/
Use scipy.optimize.check_grad. Assuming a correct evaluation of the loss function, if your gradient is implemented correctly, the output should be very small. Of course this function is very expensive, thus you should try it on a small scale data (say try it on synthetic dataset, which only has 2 features).  
3, Evaluation Metrics
Evaluate by comparing different baselines, different parameter settings. The point is to see both the accuracy and the efficiency.
1)	Accuracy: number of correctly classified test data over the whole dataset. (Function is already implemented for you.)
2)	Convergence rate: plot the loss function as a function of the number of iterations. You should see a converging curve. Try this for different initial learning rates, draw different curves.
3)	Measure the average time for each iteration of training, and for prediction (on different datasets).
4, Data
Different datasets have been provided.

