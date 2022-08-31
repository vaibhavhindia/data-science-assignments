# Data-Science-Assignments


# This Contains my notes/study material releated to the topics and Python code of all the machine learning algorithms.

# Index


1 assignment - Basic statistics(1)

2 assignment - Basic statistics(2)

3 assignment - Hypothesis Testing

4 assignment - Simple Linear Regression

5 assignment - Multiple Linear Regression

6 assignment - Logistic Regression

7 assignment - Clustering

8 assignment - PCA

9 assignment - Association Rules

10 assignment - Recommendation Engine

11 assignment - Text Mining

12 assignment - Naive Bayes

13 assignment - KNN

14 assignment - Decision Tree

15 assignment - Random Forest

16 assignment - Neural Network

17 assignment - SVM

18 assignment - Forecasting

# Training and Testing Data
 1. Its good practice to first randomly sort and the data then split into two parts. 80% of data for training the model and remaining 20% of the data for     testing the model.
 2. The reason we don't use same training set for testing is because our model has seen those samples before, using same samples for making predictions might give us wrong impression about accuracy of our model.
 3. Here we are going to use sklearn.model_selection.train_test_split method.
 
 
 # Logistic Regression
 1. Logistic regression is a supervised learning classification algorithm used to predict the probability of a target variable.
 2. Binary classficiation: When outcome has only two categories. (yea/no, 0/1, buy/not buy) e.g. Predicting whether customer will buy insurance policy.
 3. Multiclass classification: When outcome has more than two categoirs. e.g Which party a person is going to vote for (BJP, Congres, AAP).
 
 # Decisions Trees 
 1. Decisions trees are the most powerful algorithms that falls under the category of supervised algorithms.
 2. Unlike other supervised learning algorihms decision tree can be used to solve regression and classification problems.
 3. The goal of decision tree is to create training model that can predict class or value by learning simple decision rules from training data.
 4. The two main entities of a tree are decision nodes, where the data is split and leaves, where we got outcome.
 5. Decision tree algorithm forms the tree, based on 'High Information Gain'. Lower the entropy higher the information gain.
 6. Entropy: It is basically a measure of randomness in your sample. So if there is no randomness in your sample then then entropy is low.
 7. Types of decision tree
  a. Classification decision trees − In this kind of decision trees, the decision variable is categorical. The above decision tree is an example of classification decision tree.
  b. Regression decision trees − In this kind of decision trees, the decision variable is continuous.
  
  
  # Support Vector Machine (SVM)
  1. SVM algorithm is preferred by many as it provide more accurate results with less computational power.
  2. SVM are mostly used for classification tasks but can also be used for regression tasks as well.
  3. SVM is suited for extreame cases( where difference between feature is very small. e.g. cat which groomed like a Dog).
  4. So the SVM will looks at the extreame points in dataset and draws a boundary (line incase of 2D and hyperplane for more 2D) between those extreame      points to separate the features. Which results in best possible segregation of classes.
  5. Suport vectors are the data points which are close to the opposing class. SO SVM actually only consider these support vectors for defining the          classification boundary and ignore's the other training examples.
  6. e.g. suppose we have a dataset of dogs and cats. In that dataset there is a dog that looks like a cat and a cat thats is groomed like a dog. So        our SVM algorithm will use these two extreame examples as support vectors and draws boundary to classify the dogs and cats classes. Since this          boundary is based on extream examples(support vector) it will takes care of other training examples as well.
  7. SVM will use multiple such support vectors to classify dataset and increase the margin between to classes .
  
  # SVM parameters
  1. Gamma: In case of high value of Gamma decision boundary is dependent of points cloase it where in case of low value of Gamma decision SVM will          consider the far away points also while deciding the decision boundary .
  2. Regularization parameter(C): Large C will result in overfitting and which will lead to lower bias and high variance. Small C will result in            underfitting and which will lead to higher bias and low variance .
  # References
  . https://www.youtube.com/watch?v=FB5EdxAGxQg
  . https://www.youtube.com/watch?v=Y6RRHw9uN9o
  . https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47
  . https://www.youtube.com/watch?v=m2a2K4lprQw

  

 
 
 

