# Supervised Machine Learning Part-2
This module covers more advanced supervised learning methods that include ensembles of trees (random forests, gradient boosted trees), and neural networks (with an optional summary on deep learning). You will also learn about the critical problem of data leakage in machine learning and how to detect and avoid it.

# Naive Bayes Classifiers
These classifiers are called 'Naive' because they assume that features are conditionally independent, given the class. In other words they assume that, for all instances of a given class, the features have little/no correlation with each other. There are three flavors of Naive Bayers Classifier that are available in scikit-learn:
1. Bernouli: binary features (e.g. word presence/absence)
2. Multinomial: discrete features (e.g. word counts)
3. Gaussian: continuous/real-valued features

### Gaussian Naive Bayes
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




