# Overfitting and Underfitting
### Overfitting
Overfitting happens when a model learns the detail and noise in the training data to the extent that it negatively impacts the performance of the model on new data. This means that the noise or random fluctuations in the training data is picked up and learned as concepts by the model. The problem is that these concepts do not apply to new data and negatively impact the models ability to generalize.
### Underfitting
Underfitting refers to a model that can neither model the training data nor generalize to new data. An underfit machine learning model is not a suitable model and will be obvious as it will have poor performance on the training data. Underfitting is often not discussed as it is easy to detect given a good performance metric. The remedy is to move on and try alternate machine learning algorithms. Nevertheless, it does provide a good contrast to the problem of overfitting.
<img src="https://media.geeksforgeeks.org/wp-content/uploads/20190523171704/overfitting_21.png">

# Normalization
Important for some machine learning methods that all features are on the same scale(e.g faster convergence in learning, more uniform of fair influence for all weights)
### MinMax scaling feature
<img src="https://media.geeksforgeeks.org/wp-content/uploads/min-max-normalisation.jpg">

# Linear Regression and Regularization Techniques
### Linear Regression
<img src="https://media.geeksforgeeks.org/wp-content/uploads/linear-regression-plot.jpg">
Linear regression performs the task to predict a dependent variable value (y) based on a given independent variable (x). So, this regression technique finds out a linear relationship between x (input) and y(output). Hence, the name is Linear Regression. In the figure above, X (input) is the work experience and Y (output) is the salary of a person. The regression line is the best fit line for our model.
<img src="https://media.geeksforgeeks.org/wp-content/uploads/linear-regression-hypothesis.jpg">
While training the model we are given :
x: input training data (univariate – one input variable(parameter))
y: labels to data (supervised learning)

When training the model – it fits the best line to predict the value of y for a given value of x. The model gets the best regression fit line by finding the best θ1 and θ2 values.
θ1: intercept
θ2: coefficient of x

Once we find the best θ1 and θ2 values, we get the best fit line. So when we are finally using our model for prediction, it will predict the value of y for the input value of x.

How to update θ1 and θ2 values to get the best fit line ?

Cost Function (J):
By achieving the best-fit regression line, the model aims to predict y value such that the error difference between predicted value and true value is minimum. So, it is very important to update the θ1 and θ2 values, to reach the best value that minimize the error between predicted y value (pred) and true y value (y).

<img src='https://media.geeksforgeeks.org/wp-content/uploads/LR-cost-function-1.jpg'>

### Regularization Technique
Regularization is one of the most important concepts of machine learning. This technique prevents the model from overfitting by adding extra information to it. It is a form of regression that shrinks the coefficient estimates towards zero. In other words, this technique forces us not to learn a more complex or flexible model, to avoid the problem of overfitting.

(1) Ridge Regression (L2 Normalization)<br>
<img src='https://miro.medium.com/max/1106/1*CiqZ8lhwxi5c4d1nV24w4g.png'>

(2) Lasso Regression (L1 Normalization)<br>
<img src="https://miro.medium.com/max/1094/1*tHJ4sSPYV0bDr8xxEdiwXA.png">

# Polynomial Regression
It is also called the special case of Multiple Linear Regression in ML. Because we add some polynomial terms to the Multiple Linear regression equation to convert it into Polynomial Regression. It is a linear model with some modification in order to increase the accuracy.
<img src='https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTNR4llv7l3PAZ2vAJrJWWytJ4S5SRJhkDqx52MTwjUiepuEMhxmrB_osny67DctdNmrdo&usqp=CAU'>

# Logistic Regression
Logistic regression predicts the output of a categorical dependent variable. Therefore the outcome must be a categorical or discrete value. It can be either Yes or No, 0 or 1, true or False, etc. but instead of giving the exact value as 0 and 1, it gives the probabilistic values which lie between 0 and 1. Logistic Regression is much similar to the Linear Regression except that how they are used. Linear Regression is used for solving Regression problems, whereas Logistic regression is used for solving the classification problems.
<img src="https://www.saedsayad.com/images/LogReg_1.png">

# Support Vector Machine (Linearly Separable Data and Non-Linearly Separable Data)
A Support Vector Machine (SVM) performs classification by finding the hyperplane that maximizes the margin between the two classes. The vectors (cases) that define the hyperplane are the support vectors.

### Linearly Separable Data
Linearly Separable Data is any data that can be plotted in a graph and can be separated into classes using a straight line.
<img width="150px" height="150px" src="https://miro.medium.com/max/640/1*v0OUUim9Ur14Qsb904cMDQ.png">
<img width="150px" height="150px" src="https://miro.medium.com/max/640/1*B8zpyNKq0GT_RGQpXQMEVg.png">
### Non-Linearly Seperable Data



