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
Low Precision, High Recall
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