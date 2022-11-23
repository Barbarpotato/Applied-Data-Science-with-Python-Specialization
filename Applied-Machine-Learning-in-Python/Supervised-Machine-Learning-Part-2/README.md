# Supervised Machine Learning Part-2
This module covers more advanced supervised learning methods that include ensembles of trees (random forests, gradient boosted trees), and neural networks (with an optional summary on deep learning). You will also learn about the critical problem of data leakage in machine learning and how to detect and avoid it.

# Naive Bayes Classifiers
These classifiers are called 'Naive' because they assume that features are conditionally independent, given the class. In other words they assume that, for all instances of a given class, the features have little/no correlation with each other. There are three flavors of Naive Bayers Classifier that are available in scikit-learn:
1. Bernouli: binary features (e.g. word presence/absence)
2. Multinomial: discrete features (e.g. word counts)
3. Gaussian: continuous/real-valued features

How it Works:<br>
<img src='https://2.bp.blogspot.com/-sD_VfJzi8YY/WtTygMEGRCI/AAAAAAAABwA/mnnX-Q14j3kRoFzbygUrhgDS_DQwSemZQCLcBGAs/s640/Decision%2BTree%2BExercise.jpg'><br>
The dataset is divided into two parts, feature matrix and the response vector.
- Feature matrix contains all the vectors(rows) of dataset in which each vector consists of the value of - dependent features. In above dataset, features are ‘Outlook’, ‘Temperature’, ‘Humidity’ and ‘Windy’.
- Response vector contains the value of class variable(prediction or output) for each row of feature matrix. In above dataset, the class variable name is ‘Play golf’.<br>

<img src='https://media.geeksforgeeks.org/wp-content/uploads/naive-bayes-classification.png'><br>
So, in the figure above, we have calculated P(xi | yj) for each xi in X and yj in y manually in the data tables or example, probability of playing golf given that the temperature is cool, i.e P(temp. = cool | play golf = Yes) = 3/9. Also, we need to find class probabilities (P(y)) which has been calculated in the table 5. For example, P(play golf = Yes) = 9/14.

So now, we are done with our pre-computations and the classifier is ready! Let us test it on a new set of features (let us call it today):<br>
today = (Sunny, Hot, Normal, False)

Probability of playing golf is given by:
2/9 * 2/9 * 6/9 * 6/9 * 9/14 = 0.0141<br>
Probability of not playing golf given by:
3/5 * 2/5 * 1/5 * 2/5 * 5/14 = 0.0068<br>

These numbers can be converted into a probability by making the sum equal to 1 (normalization):
P(YES | today ) = 0.00141 / (0.0141 + 0.0068) = 0.67
P(NO | today) = 0.0068 / (0.0068 + 0.0141) = 0.33

### Gaussian Naive Bayes
This Dataset is used for the example of how gaussian naive bayes works, contain 7 data from cancer class
and 7 data from healthy class, with 1 feature is PSA.<br>
<img src='https://github.com/Barbarpotato/Applied-Data-Science-with-Python-Specialization/blob/main/Applied-Machine-Learning-in-Python/Supervised-Machine-Learning-Part-2/images/data-gaussian.png'><br>

for the Gaussian Concept, we need to calculate the mean of some feature in class, and its standart deviation.
for cancer we got std=0.82 and the mean=2.8, for healthy class we got std=0.61 and mean=1.5<br>
<img src='https://github.com/Barbarpotato/Applied-Data-Science-with-Python-Specialization/blob/main/Applied-Machine-Learning-in-Python/Supervised-Machine-Learning-Part-2/images/mean-std-gaussian.png'><br>

After we got the std and the mean of each class, we can calculate the f(x) of PSA feature that we want to predict. for this example we use PSA = 2.6<br>
<img src='https://github.com/Barbarpotato/Applied-Data-Science-with-Python-Specialization/blob/main/Applied-Machine-Learning-in-Python/Supervised-Machine-Learning-Part-2/images/calculate-gaussian.png'><br>

Do not forget to calculate the probability of cancer and healthy in the dataset. in this case we have 7 cancer and 7 healthy dataset which made the p(Cancer) and the p(Healthy) are 0.5<br>
<img src='https://github.com/Barbarpotato/Applied-Data-Science-with-Python-Specialization/blob/main/Applied-Machine-Learning-in-Python/Supervised-Machine-Learning-Part-2/images/proba-gaussian.png'><br>

then count the probability where the PSA = 2.6, in this case we got 79% chance diagnosed as Cancer and its about 21% chance healthy when the PSA = 2.6<br>
<img src='https://github.com/Barbarpotato/Applied-Data-Science-with-Python-Specialization/blob/main/Applied-Machine-Learning-in-Python/Supervised-Machine-Learning-Part-2/images/proba2-gaussian.png'><br>

# Ensamble Learning 
Ensemble learning, in general, is a model that makes predictions based on a number of different models. By a combining a number of different models, an ensemble learning tends to be more flexible (less bias) and less data sensitive (less variance). The two most popular ensemble learning methods are bagging(random forest) and boosting. Bagging : Training a bunch of models in parallel way. Each model learns from a random subset of the data. Boosting : Training a bunch of models sequentially. Each model learns from the mistakes of the previous model.
<img src="https://www.machinelearningplus.com/wp-content/uploads/2020/11/output_12_0-1.png">

### Random Forest
Decision Tree are highly sensitive to the training data which could result in high variance. So Our Model might failed to generalize, so here comes the Random Forest algorithm. Random forests or random decision forests is an ensemble learning method for classification, regression and other tasks. It is a collection of multiple random decision trees and its much less sensitive to the training data. The Step of Random FOrest Algrithm:
1. build the datasets from the original data.
in this case, the data will get randomed select from the original dataset. Every dataset will contain the same number of rows as the original one. Not just randomly picked the dataset, we'll randomly select the subset of features as well for each tree and use it for training. in this example we will use 4 random sampling. The Process when we created the new dataset are:
- 'Bootstrapping'. Bootstraping ensures that we are not using the same data for every tree, so it helps our model to be less sensitive to the original dataset.
- 'Random feature selection'. helps to reduce the correlation between the trees.if we use every feautre then most of our trees will have the same decision nodes and they will act very similiarly.
<img src='https://github.com/Barbarpotato/Applied-Data-Science-with-Python-Specialization/blob/main/Applied-Machine-Learning-in-Python/Supervised-Machine-Learning-Part-2/images/randomforest-data-split.png'>

2. creating the decision tree each of the new dataset.
in this case we have 4 new dataset that generated using the Bootstrapping method. and each of the dataset contain different features. Create the decision tree each of this dataset and the result will look like this:<br>
<img src='https://github.com/Barbarpotato/Applied-Data-Science-with-Python-Specialization/blob/main/Applied-Machine-Learning-in-Python/Supervised-Machine-Learning-Part-2/images/dec-tree-randomforest.png'>

3. Predicting some values.
after we build each of the decision tree, lets take a test data for our model. In this case we are gonna use 2.8, 6.2, 4.3, 5.3, 5.5. this data will be tested in 4 different decision tree models and generated the output between one and zero (binary classification problem) which: 1, 0, 1, 1. From this output we will take the majority vote which is predicted as 1. The process of combining result between different decision tree models is called 'aggregation'.

### Gradient Boosting Decision Trees
In gradient boosting decision trees, we combine many weak learners to come up with one strong learner. The weak learners here are the individual decision trees. All the trees are conncted in series and each tree tries to minimise the error of the previous tree. Due to this sequential connection, boosting algorithms are usually slow to learn, but also highly accurate. In statistical learning, models that learn slowly perform better.
<img src="https://www.machinelearningplus.com/wp-content/uploads/2020/11/output_20_0.png">
The weak learners are fit in such a way that each new learner fits into the residuals of the previous step so as the model improves. The final model aggregates the result of each step and thus a strong learner is achieved. A loss function is used to detect the residuals. For instance, mean squared error (MSE) can be used for a regression task and logarithmic loss (log loss) can be used for classification tasks. It is worth noting that existing trees in the model do not change when a new tree is added. The added decision tree fits the residuals from the current model.

# Multi Layer Perceptron
<img src="https://github.com/Barbarpotato/Applied-Data-Science-with-Python-Specialization/blob/main/Applied-Machine-Learning-in-Python/Supervised-Machine-Learning-Part-2/images/Multi-layer-perceptron.png">
A multilayer perceptron (MLP) is a fully connected class of feedforward artificial neural network (ANN). An MLP consists of at least three layers of nodes: an input layer, a hidden layer and an output layer. Except for the input nodes, each node is a neuron that uses a nonlinear activation function. MLP utilizes a supervised learning technique called backpropagation for training.

# Data Leakage
Data leakage can cause you to create overly optimistic if not completely invalid predictive models. Data leakage is when information from outside the training dataset is used to create the model. This additional information can allow the model to learn or know something that it otherwise would not know and in turn invalidate the estimated performance of the mode being constructed.
### Detecting Data Leakage
* Before Building the model
    - Exploratory data analysis to find surprises in the data.
    - Are there features very highly correlated with the target value?

* After Building the model
    - Look for surprising feature behavior in the fitted model.
    - Are there features with very high weights, or high information gain?

* Limited real-world deployment of the trained model
    - Potentially expensive in terms of development time, but more realistic.
    - is the trained model generalizing well to new a data?

### Minimizing Data Leakage
* Perform data preparation within each cross-validation fold seperately
    - Scale/Normalize data, perform feature selection,etc. within each fold seperately, not using the entire dataset.
    - For any such parameters estimated on the training data, you must use those same parameters to prepare data on the corresponding held-out test fold.
* With time series data, use a timestamp cut off
    - the cutoff value is set to the specific time point where prediction is to occur using current and past records.
    - Using the cutoff time will make sure you are not accessing any data records that were gathered after the prediction time, i.e in the future.
* Before any work with a new dataset, split off a final test validation dataset
    - ... if you have enough data.
    - Use this final test dataseet as the very last step in your validation.
    - Helps to check the true generalization performance of an trained models. 
