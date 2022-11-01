# Supervised Machine Learning Part-1
This module delves into a wider variety of supervised learning methods for both classification and regression, learning about the connection between model complexity and generalization performance, the importance of proper feature scaling, and how to control model complexity by applying techniques like regularization to avoid overfitting. In addition to k-nearest neighbors, this week covers linear regression (least-squares, ridge, lasso, and polynomial regression), logistic regression, support vector machines, the use of cross-validation for model evaluation, and decision trees.

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

# Support Vector Machine (Linearly Separable Data and Non-Linearly Separable Data Transformation) and Tuning Parameters
A Support Vector Machine (SVM) performs classification by finding the hyperplane that maximizes the margin between the two classes. The vectors (cases) that define the hyperplane are the support vectors.

### Linearly Separable Data
Linearly Separable Data is any data that can be plotted in a graph and can be separated into classes using a straight line.<br>
<p float='left'>
<img width='300px' height='250px' src="https://miro.medium.com/max/640/1*v0OUUim9Ur14Qsb904cMDQ.png"><br>
<img width='300px' height='250px' src="https://miro.medium.com/max/640/1*B8zpyNKq0GT_RGQpXQMEVg.png">
</p>

### Non-Linearly Seperable Data
In this case we cannot find a straight line to separate<br>
<img src="https://i.stack.imgur.com/y5uMX.png"><br>
<br>
The basic idea is that when a data set is inseparable in the current dimensions, add another dimension, maybe that way the data will be separable. Just think about it, the example above is in 2D and it is inseparable, but maybe in 3D there is a gap between them, maybe there is a level difference. In this case, we can easily draw a separating hyperplane (in 3D a hyperplane is a plane) between level 1 and 2.

Let's assume that we have 2 Dimensional data, but its not separable, so we add another dimension called X3. Another important transformation is that in the new dimension the points are organized using this formula x1² + x2².

If we plot the plane defined by the x² + y² formula, we will get something like this:
<img src="https://i.stack.imgur.com/oiZYz.jpg"><br>

These transformations are called kernels.
Popular kernels are: Polynomial Kernel, Gaussian Kernel, Radial Basis Function (RBF), Laplace RBF Kernel, Sigmoid Kernel, Anove RBF Kernel, etc

(1) Radial Basis Function Kernel Transformation
<br>
<img src='https://miro.medium.com/max/530/1*ZMCGXM4ROxEXlNe0SUlToA.jpeg'><br>
where,<br>
1. ‘σ’ is the variance and our hyperparameter
2. ||X₁ - X₂|| is the Euclidean (L₂-norm) Distance between two points X₁ and X₂

(2) Polynomial Kernel<br>
K(x, y) = (x^T.y + c)^d<br>
where x and y are vectors in the input space, i.e. vectors of features computed from training or test samples and c ≥ 0 is a free parameter trading off the influence of higher-order versus lower-order terms in the polynomial. When c = 0, the kernel is called homogeneous. (A further generalized polykernel divides xTy by a user-specified scalar parameter a.)

### Tuning Parameters.
(1) Regularization<br>
The Regularization Parameter (in python it’s called C) tells the SVM optimization how much you want to avoid miss classifying each training example.

If the C is higher, the optimization will choose smaller margin hyperplane, so training data miss classification rate will be lower.
<img src="https://miro.medium.com/max/828/0*rvt2H-wO55hKjJ5Y.png"><br>

(2) Gamma<br>
The next important parameter is Gamma. The gamma parameter defines how far the influence of a single training example reaches. This means that high Gamma will consider only points close to the plausible hyperplane and low Gamma will consider points at greater distance.<br>
<img src="https://miro.medium.com/max/828/0*P5cqyr_n84SQDuAN.png"><br>

(3) Margin<br>
The last parameter is the margin. We’ve already talked about margin, higher margin results better model, so better classification (or prediction). The margin should be always maximized.

# Cross Validation
### K-Fold Cross Validation
K-fold Cross-Validation is when the dataset is split into a K number of folds and is used to evaluate the model's ability when given new data. K refers to the number of groups the data sample is split into. For example, if you see that the k-value is 5, we can call this a 5-fold cross-validation.
<img src='https://cdn-images-1.medium.com/max/1009/1*1RPHQk-xpKMInxkEd1qFyg.png'>

### Stratified Cross Validation
Example Explanation:
Let’s consider a binary-class classification problem. Let our dataset consists of 100 samples out of which 80 are negative class { 0 } and 20 are positive class { 1 }.

If we do random sampling to split the dataset into training_set and test_set in an 8:2 ratio respectively.Then we might get all negative class {0} in training_set i.e 80 samples in training_test and all 20 positive class {1} in test_set.Now if we train our model on training_set and test our model on test_set, Then obviously we will get a bad accuracy score.

In stratified sampling, The training_set consists of 64 negative class{0} ( 80% 0f 80 ) and 16 positive class {1} ( 80% of 20 ) i.e. 64{0}+16{1}=80 samples in training_set which represents the original dataset in equal proportion and similarly test_set consists of 16 negative class {0} ( 20% of 80 ) and 4 positive class{1} ( 20% of 20 ) i.e. 16{0}+4{1}=20 samples in test_set which also represents the entire dataset in equal proportion.This type of train-test-split results in good accuracy.

### Leave-One-Out Cross Validation
In the leave-one-out (LOO) cross-validation, we train our machine-learning model n times where n is to our dataset’s size. Each time, only one sample is used as a test set while the rest are used to train our model.

<img src='https://www.baeldung.com/wp-content/uploads/sites/4/2022/05/loso.png'><br>
The final performance estimate is the average of the six individual scores:<br>
<h5>Overall Score = (Score1+ Score2 + Score3 + Score4 + Score5 + Score6) / 6</h5>

# Decision Tree
A decision tree is a tree-like structure that is used as a model for classifying data. A decision tree decomposes the data into sub-trees made of other sub-trees and/or leaf nodes.
Step By Step:<br>
<img src='https://2.bp.blogspot.com/-sD_VfJzi8YY/WtTygMEGRCI/AAAAAAAABwA/mnnX-Q14j3kRoFzbygUrhgDS_DQwSemZQCLcBGAs/s640/Decision%2BTree%2BExercise.jpg'><br>

### Determine The Decision Column
We Can calculate how many data in some class.<br>
<img src='https://3.bp.blogspot.com/-sr5Xk0iBLZM/WtUToEVlKSI/AAAAAAAABwQ/914mIDeieOUpVG38pYwx3Q1uVkOBYYXRwCLcBGAs/s200/Decistion%2BTree%2B-%2BFrequency%2BTable%2B-%2BPlay%2BGolf.jpg'><br>

### Calculating the Entropy for the classes
<img src='https://2.bp.blogspot.com/-nCz0cZ8jYMQ/WtUWR1NJXdI/AAAAAAAABww/qdjyvECbSr4IiBSpYCevuznnKcNNjHmSgCLcBGAs/s400/Decistion%2BTree%2B-%2BEntropy%2BCalculation.jpg'><br>

### Calculating Entropy for other attributes
For the other four attributes, we need to calculate the entropy after each of the split.<br>
<ul>
<li>(PlayGolf, Outloook)</li>
<li>(PlayGolf, Temperature)</li>
<li>(PlayGolf, Humidity)</li>
<li>(PlayGolf,Windy)</li>
</ul>
Example of (PlayGolf, Outlook):<br>
<img src='https://4.bp.blogspot.com/-R2Y6cMoCA2I/WtUh53NPSxI/AAAAAAAABxY/zxCy_7Iz8gI3ha0sthZt_7nvgat42G2-ACLcBGAs/s320/Decistion%2BTree%2B-%2BEntropy%2Bof%2BTwo%2BVariables3.jpg'>
<p>Using this table, we can then calculate E(PlayGolf, Outlook), which would then be given by the formula below</p>
<img src='https://2.bp.blogspot.com/-imdc1oWPMe8/WtUj2JtJNzI/AAAAAAAABxs/ch4jn3jU-2UrzM7vgoWOVihVBEhyPsSuQCLcBGAs/s400/Decistion%2BTree%2B-%2BEntropy%2Bof%2BTwo%2BVariables4.jpg'><br>
Let’s go ahead to calculate E(3,2)
We would not need to calculate the second and the third terms! This is because
E(4, 0) = 0
E(2,3) = E(3,2)<br>
<img src='https://2.bp.blogspot.com/-HsYFjNR0xdI/WtUma2vNVHI/AAAAAAAABx4/IV0N_y8VlvodugKhxTyaCIatgYVxdRNrQCLcBGAs/s400/Decistion%2BTree%2B-%2BEntropy%2Bof%2BTwo%2BVariables5.jpg'><br>
and the result of E(PlayGolf, Outloook) is:
<br>
<img src='https://4.bp.blogspot.com/-DK4jUKpsE5A/WtZVqiOqUFI/AAAAAAAAByc/KaXYSR50STUYeQ8fwGDiExQEs7CY59PagCLcBGAs/s400/Calculating%2BP%2528PlayGolf%252C%2BOutlook%2529.jpg'><br>
After Calculating all attributes, we have:<br>
<ol>
<li>E(PlayGolf, Outloook) = 0.693</li>
<li>E(PlayGolf, Temperature) = 0.911</li>
<li>E(PlayGolf, Humidity) = 0.788</li>
<li>E(PlayGolf,Windy) = 0.892</li>
</ol>

### Calculating Information Gain for Each Split
The next step is to calculate the information gain for each of the attributes. The information gain is calculated from the split using each of the attributes. Then the attribute with the largest information gain is used for the split.
The information gain is calculated using the formula:<br>
<br><strong>Information Gain: Entropy class - Entropy Attributes</strong>
<br><br>
So let’s go ahead to do the calculation:<br>

Gain(PlayGolf, Outlook) = Entropy(PlayGolf) – Entropy(PlayGolf, Outlook)
= 0.94 – 0.693 = 0.247

Gain(PlayGolf, Temperature) = Entropy(PlayGolf) – Entropy(PlayGolf, Temparature)
= 0.94 – 0.911 = 0.029

Gain(PlayGolf, Humidity) = Entropy(PlayGolf) – Entropy(PlayGolf, Humidity)
= 0.94 – 0.788 = 0.152

Gain(PlayGolf, Windy) = Entropy(PlayGolf) – Entropy(PlayGolf, Windy)
= 0.94 – 0.892 = 0.048

### Perform the First Split
Now that we have all the information gain, we then split the tree based on the attribute with the highest information gain.

### Controlling the Model Complexity of Decision Tree
<img src='https://4.bp.blogspot.com/-pPu1zJ4iKgk/XdYjR_9C_jI/AAAAAAAAA98/-YCd2tTxGlIfApkrlmmatwIo-eVVgnrIwCLcBGAsYHQ/s1600/neuralnetworklayersbeforeandafterpruning.png'>
<p>This technique is used after construction of decision tree. This technique is used when decision tree will have very large depth and will show overfitting of model. We can make it to be a simple form by using post-pruning technique.</p>

