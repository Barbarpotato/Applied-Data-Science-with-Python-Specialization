# Evaluation and Model Selection 
This module covers evaluation and model selection methods that you can use to help understand and optimize the performance of your machine learning models.

# Imbalanced Dataset
An imbalanced classification problem is an example of a classification problem where the distribution of examples across the known classes is biased or skewed. The distribution can vary from a slight bias to a severe imbalance where there is one example in the minority class for hundreds, thousands, or millions of examples in the majority class or classes.

This results in models that have poor predictive performance, specifically for the minority class. This is a problem because typically, the minority class is more important and therefore the problem is more sensitive to classification errors for the minority class than the majority class.

We can use sanity check on our classifier perfomance for imbalanced datasets problems using:
### Dummy Classifiers
This classifier will provide a null metric(e.g null accuracy) baseline. that means the accuracy can be achieved by always picking the most frequent class. this classifiers doesnt use for real problems. this classifiers is useed to check the sanity on our real classifiers performance.
Some commonly-used settings for the strategy parameter for DummyClassifier in Scikit-learn:
-most_frequent: predict the most frequent label in the training set.
-stratified: random predictions based on training set class distributions.
-uniform: generates predictions uniformly at random. all class have an equally chances at being output.
-constant: always predict a constant label provided by a user.
### Dummy Regressors
This is used for the regression problem
strategy parameter options:
-mean: predicts the mean of the training targets
-median: predicts the median of the training targets.
-quantile: predicts a user-provided quantile of the training targets
-constant: predicts a constant user-provided value.

# Confusion Matrices & Basic Evaluation Metrics
A confusion matrix of binary classification is a two by two table formed by counting of the number of the four outcomes of a binary classifier. We usually denote them as TP, FP, TN, and FN instead of “the number of true positives”, and so on.
<img src='https://miro.medium.com/max/1000/1*3_ymeNrZayqAcuvJcSZk_Q.png'>
<br>

### Accuracay
For what fraction of all instances is the classifier's prediction correct?
<p><strong>True Positive + True Negative / True Negative + True Positives + False Positives + False Negatives</strong></p>

### Clasification Error
For what fraction of all instances is the classifiers prediction incorrect?
<p><strong>False Positive + False Negative / True Negative + True Positives + False Positives + False Negatives</strong></p>

### Recall
For what fraction of all positives instances does the clasifier correctly identify as positive?
<p><strong>True Positive / True Positives + False Negatives</strong></p>

### Precision
what fraction of positive predictions are correct?
<p><strong>True Positive / True Positives + False Positive</strong></p>

### There is often a tradeoff between precision and recall
Low Precision, High Recall<br>
High Precision, Low Recall

Recall-Oriented machine learning tasks:
1. Search and information extraction in legal discovery
2. Tumor Detection
3. Often paired with a human expert to filter out false positive

Precision-Oriented machine learning tasks:
1. Search Engine ranking, query suggestion
2. Document Classification
3. Many customer-facing tasks

### F-1 Score
Combining precision and recall into a single number.
where, 
F1 = 2*TP / (2*TP+FN+FP)

# Classifier Decision Function
This method basically returns a Numpy array, In which each element represents whether a predicted sample for x_test by the classifier lies to the right or left side of the Hyperplane and also how far from the HyperPlane. It also tells us that how confidently each value predicted for x_test by the classifier is Positive ( large-magnitude Positive value ) or Negative ( large-magnitude Negative value)

### Decision Threshold
<img width='800px' height='700px' src='https://github.com/Barbarpotato/Applied-Data-Science-with-Python-Specialization/blob/main/Applied-Machine-Learning-in-Python/Evaluation-and-Model-Selection/Images/decision_threshold.png'><br>
sklearn does not let us set the decision threshold directly, but it gives us the access to decision scores ( Decision function o/p ) that is used to make the prediction. We can select the best score from decision function output and set it as Decision Threshold value and consider all those Decision score values which are less than this Decision Threshold as a negative class ( 0 ) and all those decision score values that are greater than this Decision Threshold value as a positive class ( 1 ).

1. Precision-Recall Curves<br>
Using Precision-Recall curve for various Decision Threshold values, we can select the best value for Decision Threshold such that it gives High Precision ( Without affection Recall much ) or High Recall ( Without affecting Precision much ) based on whether our project is precision-oriented or recall-oriented respectively.<br>
<img width='800px' height='700px' src='https://github.com/Barbarpotato/Applied-Data-Science-with-Python-Specialization/blob/main/Applied-Machine-Learning-in-Python/Evaluation-and-Model-Selection/Images/precision-recall-curves.png'><br>

2. ROC (Reveicer Operating Characteristic) Curves<br>
ROC curve is a graph that shows the performance of a classification model at all possible thresholds( threshold is a particular value beyond which you say a point belongs to a particular class). The curve is plotted between two parameters: TRUE POSITIVE RATE and FALSE POSITIVE RATE.
<img width='800px' height='700px'  src='https://github.com/Barbarpotato/Applied-Data-Science-with-Python-Specialization/blob/main/Applied-Machine-Learning-in-Python/Evaluation-and-Model-Selection/Images/roc-curves.png'>
