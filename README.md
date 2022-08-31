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
  
  ## SVM parameters
  1. Gamma: In case of high value of Gamma decision boundary is dependent of points cloase it where in case of low value of Gamma decision SVM will          consider the far away points also while deciding the decision boundary .
  2. Regularization parameter(C): Large C will result in overfitting and which will lead to lower bias and high variance. Small C will result in            underfitting and which will lead to higher bias and low variance .
  ## References
  1. https://www.youtube.com/watch?v=FB5EdxAGxQg
  2. https://www.youtube.com/watch?v=Y6RRHw9uN9o
  3. https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47
  4. https://www.youtube.com/watch?v=m2a2K4lprQw
  
  # Random Forest 
  1. Random forest is supervised learning algoriothm which is used for classification as well as regression. However it is mostly used for                  classfication problems .
  2. In Random forest algorithm dataset is devided in multiple batches and using 'Decision Tree' algorithm, gets prediction for each batch and then          choose the best solution based on voting.
  3. We can understand the working of Random Forest algorithm with the help of following steps:
    1. Step 1 − First, start with the selection of random samples from a given dataset.
    2. Step 2 − Next, this algorithm will construct a decision tree for every sample. Then it will get the prediction result from every decision tree.
    3. Step 3 − In this step, voting will be performed for every predicted result.
    4. Step 4 − At last, select the most voted prediction result as the final prediction result.
     
## Pros
1. It overcomes the problem of overfitting by averaging or combining the results of different decision trees.
2. Random forests work well for a large range of data items than a single decision tree does.
3. Random forest has less variance then single decision tree.
4. Random forests are very flexible and possess very high accuracy.
5. Scaling of data does not require in random forest algorithm. It maintains good accuracy even after providing data without scaling.
6. Random Forest algorithms maintains good accuracy even a large proportion of the data is missing.

## Cons
1. Complexity is the main disadvantage of Random forest algorithms.
2. Construction of Random forests are much harder and time-consuming than decision trees.
3. More computational resources are required to implement Random Forest algorithm.
4. It is less intuitive in case when we have a large collection of decision trees.
5. The prediction process using random forests is very time-consuming in comparison with other algorithms.

# K Means Clustering
1. K Means is unsupervised learning algorith. It is used to find the clusters of data in unlabelled data.
2. K = No of principal componenet or no of clusters.

## How K Means Algorith works
1. First steps is to randomly initialize two points and call them centroids .
2. No of centroids should be equal to no of clusters you want to predict.
3. Now in 'assignment steps' K Means algorithm will go through each of the data points and depending on its closeness to the cluster it will assign      the data points to a cluster.
4. During 'assignment' if there is any centroid who has no data point associated with it, then it can be removed.
5. Now in 'move' step K means algorithm will find the mean of each data point assigned to the cluster centroid and move the respective centroid to the    mean value location.
6. Now alogorith will keep doing the 'assigment' and 'move' steps till the convergance.

## Choosing no of clusters (k)
1. Mostly K value choosen mannually
2. Elbow Method

# Naive Bayes Algorithm
## Basic Probability
1. Probability of getting head/tail when you flip a coin is 0.5 i.e. 50% .
2. Similarly probability of getting queen from a deck of card is 4/52 i.e. 7.7 %

## Consitional Probability
1. Unlike basic probability in conditional probability we know that the event A has occured and we are trying to predict the probability of B. i.e.      What is probability of getting a queen of diamond. Here card type diamond is event A.
2. So the consitional probability of getting a queen of diamond is represented as P(Queen/Diamond) = 1/13 i.e. 7.7%
3. More general representation is P(A/B) = Probability of 'Event A' knowing that 'Event B' has already occured .
4. Thomas Bayes conditional probability equation is: P(A/B) = ( P(B/A) * P(A) ) / P(B)

## Naive Bayes
1. So using Bayes conditional probability equation we can find the probability of certain events based on probability of some knwon events.
2. Its called 'Naive' because it assumes the known events(features) are independent of each other. This makes our calculations little simpler .

## Naive Base Classifiers

### Bernoulli Naive Bayes
1. It Assumes that all our features are binary, means they take only two values 0 and 1 .
2. e.g. 1 can represent spam mails where 0 can represent ham mails .

### Multinomial Naive Bayes
It is used when we have descrete data e.g. Movie rating from 1 to 5 as each rating will have certain frequency to represent .

### Gaussian Naive Bayes
1. Because of the assumtion of nominal distributions(bell curve) Gaussian Naive Bayes is used when all the features are continous .
2. E.g. IRIS flower dataset features(sepal width, sepal length, petal width, patal length) are continuous. We can t represent these features in terms    of their occurance which means data is continuous .

## Where its used
1. Email spam detection
2. Handwritten digit recognition
3. Weather prediction 
4. Face detection
5. News article categorization












  

  

 
 
 

