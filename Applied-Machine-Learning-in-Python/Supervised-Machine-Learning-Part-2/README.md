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
In statistics and machine learning, ensemble methods use multiple learning algorithms to obtain better predictive performance than could be obtained from any of the constituent learning algorithms alone. Ensembles combine multiple hypotheses to form a (hopefully) better hypothesis. The term ensemble is usually reserved for methods that generate multiple hypotheses using the same base learner. The broader term of multiple classifier systems also covers hybridization of hypotheses that are not induced by the same base learner.

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
