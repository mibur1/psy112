---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
myst:
  substitutions:
    ref_test: 1
---

# <i class="fa-solid fa-dice"></i> Resampling Strategies

As neuropsychologists, you will be well aware of the challenges involved in data collection — time, cost, and the complexities of experimental design often make large datasets hard to come by. However, robust predictive modeling is critical not only because extensive datasets can be rare, but also because ensuring that models generalize well to new data is often an essential question.

Resampling methods offer a powerful approach to assess model performance and mitigate overfitting. Rather than relying on a single train-test split, which can yield performance estimates that vary significantly depending on the split, resampling techniques repeatedly draw samples from your data. This process simulates multiple independent training and test sets, providing a more stable and reliable evaluation of your model.


```{admonition} Resampling Strategies
:class: hint

The two most widely used resampling methods are:

- *Cross validation*: Creating non-overlapping subsets for training and testing
- *Bootstrapping*: Sampling with replacement, resulting in (partly) overlapping samples
```

## The data

We will use a dataset you are already familiar with from last semester: The [Iris](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html) dataset which contains 150 samples from three species of the iris plant (iris setosa, iris virginica and iris versicolor). Four features were measured: the length and the width of the sepals and petals, in centimeters.

```{code-cell} 
import seaborn as sns
import pandas as pd
from sklearn import datasets

# Get data
iris = datasets.load_iris(as_frame=True)
df = iris.frame
df['class'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Plot data
sns.scatterplot(data=df, x='sepal length (cm)', y='sepal width (cm)', hue="class");
```

The goal of our model will be to classify the flowering plants based on two features shown in the plot (sepal length and width). Which of the following is true about the model and task at hand?

```{code-cell} ipython3
:tags: [remove-input]
from jupyterquiz import display_quiz
display_quiz('quiz/iris.json')
```

## Cross Validation




### Validation Set approach

```{margin}
Hyperparameter are paramateres that are not learned from the data but set by the scientist before the training process begin. T
```

In the simplest approach of cross-validation the dataset is **randomly split into two independent subsets**:
- *Training Set*: Used to train the model by learning patterns and relationships in the data
- *Validation Set*: Used to assess model performance across different models and hyperparameter choices. It therefore provides an estimation of the test error.


```{figure} figures/ValidationSet.drawio.png
:name: VS
:alt: Validation Set approach
:align: center

The validation set splits the dataset into a training and a testing set (these do not necessarily need to be of equal size).
```


Lets try this with our Iris dataset.

1.  As a first step, we need to **define the Target(y) and Features(X)**.

```{code-cell}
# thanks to Scikit-Learn, the Iris dataset is already predefined and consists of 
# defined Features and Target, which we now can use 
X, y = datasets.load_iris(return_X_y=True) 
```
2. **Split** the data into training and test sample

```{code-cell}
from sklearn.model_selection import train_test_split

# sample  a training set while holding out 40% of the data for testing the classifier
X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.4, random_state=0)
```
**Hands on**: 
Please split the data into training and test sample. The test sample should contain 73% of the data.
<iframe src="https://trinket.io/embed/python3/39cc2749b878" width="100%" height="356" frameborder="0" marginwidth="0" marginheight="0" allowfullscreen></iframe>


3. **Fit** the model
As we predict quantitaive outcome, we need a classification model. The model learns a decision boundary to separate classes in the feature space.

```{margin}
A high C will try to classify all training points correctly and leads to a complex decision boundary.
```

```{code-cell}
from sklearn import svm

model = svm.SVC(kernel='linear', C=1)
fit = model.fit(X_train, y_train)
```

4. **Evaluate** model performance
```{code-cell}
fit.score(X_test, y_test)

```
In the case of classification, we evaluate model performance using accuracy. An accuracy of ~97% means that, on average, the model correctly predicts 97 out of 100 test samples.


```{code-cell} ipython3
:tags: [remove-input]

from jupytercards import display_flashcards
display_flashcards('quiz/validation_set.json')

```

```{admonition} Summary
:class: hint

While the Validation Set Approach is a quick and easy way to check how well a model performs, it has a major flaw: it puts all its trust in a **single data split**. Imagine training a model, evaluating it once, and calling it a day — what if that split was just lucky (or unlucky)? A bad split can doom a great model, while a lucky one might trick us into thinking a weak model is stronger than it is.
```

### k-fold Cross Validation
To get a more reliable and robust performance estimate, we need something smarter— something that doesn’t leave our results up to chance. Rather than worrying about which split of data to use for training versus validation, we'll use them all in turn.

In k-fold cross validation we randomly dive the dataset into k equal-sized folds. One fold is designated at the validation set, while the remaining $k-1$ samples are the training sets. The fitting process is repeated $$k$-times, each time using a different fold as the validation set. At the end of the process, we compute the average score across all validation sets to obtain a more reliable estimate of the model's overall performance.


```{figure} figures/CV.drawio.png
:name: CV
:alt: Cross validation
:align: center

K-fold cross validation splits the dataset into $k$ equally sized parts and then trains the model on all posible combinations of it, keeping the proportion of train/test data constant.
```


Let`s also try this method on our dataset. 


1. Defining Target and Features
```{code-cell}
X, y = datasets.load_iris(return_X_y=True) 
```

```{margin}
k is also a hyperparameter!
```

2. Choosing *k*
```{code-cell}
from sklearn.model_selection import KFold
# Lets just start with k=5
k_fold = KFold(n_splits = 5)
```

3. Choosing and creating a model 


TO MICHA: Here, we could also use a DecisionTreeClassifier since it is computationally cheaper, making it more suitable for multiple repetitions, as required in K-Fold. Also for SVM we need to loop through each iteration to get the training data set as an input parameter from SVM. However, introducing two classifiers might be quite confusing for them! Especially if the exercise will then ask for regression models 

If we choose **Decision Tree**:
While we've previously used Support Vector Machines (SVMs), we might also consider a Decision Tree Classifier for K-Fold Cross-Validation. Decision Trees are computationally cheaper, making them more efficient for multiple training iterations, as required in K-Fold. 

```{code-cell}
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=42) 
```

4. Evaluate the model
```{code-cell}
from sklearn.model_selection import cross_val_score

scores = cross_val_score(clf, X, y, cv = k_fold) 

# print the score for each iteration and the average score over all folds
print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
```
The averaged model accuracy is about 91%. 

3. Choosing, creating and evaluation the model


If we choose **SVM**
As introduced in the Validation Set Approach, we can once again use a Support Vector Machine (SVM) Classifier. However, unlike the Validation Set Approach, where we have a single fixed training split, the K-Fold method dynamically changes the training and validation sets in each iteration.

```{code-cell}
from sklearn import svm

# Initialize model
clf = svm.SVC(kernel='linear', C=1)

# list to store the scores
scores=[]

# Iterate over each fold and dynamically define the training set 
for train_index, val_index in k_fold.split(X):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # Train model on current fold
    clf.fit(X_train, y_train)

    # Evaluate on validation set
    score = clf.score(X_val, y_val)
    scores.append(score)                     # Append score to list
    print(f"Validation Score : {score}")

# print average CV score
print("Average CV Score: ", sum(scores) / len(scores))
```
The averaged model accuracy is about 95%. 

```{admonition} Validation Set vs. k-fold Cross Validationach approach
:class: note

Comparing these 2 Cross Validation approaches, it can be observed that the validation set approach seems to show a better accuracy. But does this reflects the reality?
- The Validation Set Approach often **overestimates** accuracy since it tests the model on a single split, leading to potential overfitting. 
- In contrast, K-Fold Cross-Validation provides a **more reliable** assessment by evaluating multiple splits, reducing bias, and offering a realistic performance estimate. 
- If K-Fold accuracy is significantly lower, it indicates that the initial estimate was overly optimistic. 

Therefore, K-Fold is preferred as it provides a more trustworthy measure of real-world performance.
```

**Hands on**:
Please change k. 

<iframe src="https://trinket.io/embed/python3/29cdfb10f7f9" width="100%" height="356" frameborder="0" marginwidth="0" marginheight="0" allowfullscreen></iframe>


```{admonition} The big choice of k
:class: note 

Choosing an appropriate k involves a trade-off between bias, variance, and computational cost.
A higher k generally provides a more stable and reliable estimate but comes with higher computational effort, making it often impractical in real-world applications.

Generally speaking, **k = 5 or k = 10** is most common, as it balances computation time and reliability.
```


```{margin}
In LOOCV k=n!
```

### Leave-one-out Cross Validation (LOOCV)
LOOCV is a special case of K-Fold Cross-Validation, where k equals the number of observations. In LOOCV, the model is trained on all but one data point, and the remaining single observation is used for validation. This process repeats for each data point, ensuring every observation is used for testing exactly once. 

While LOOCV provides a low-bias estimate, it is computationally expensive and may lead to high variance in model performance. 

In Python, the Leave One Out approach is very similiar to the k-fold procedure. Only the Cross validation method is changed to `LeaveOneOut()`. 

```{code-cell}
# import packages
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_score

# define target and features
X, y = datasets.load_iris(return_X_y=True)

# define classifier model
clf = DecisionTreeClassifier(random_state=42)

# define Cross Validatin approach
loo= LeaveOneOut()

# fit model
scores = cross_val_score(clf, X, y, cv = loo)

# Print scores
print("Average CV Score: ", scores.mean())
```


## **Bootstrapping**
Bootstrapping is a resampling technique in statistics and machine learning that repeatedly draws samples **with replacement** from a dataset to estimate a population parameter. It can be used to quantify the uncertainty associated with a given estimator or statistical learning method. 

As seen in Cross-Validation, Bootstrapping also uses training and validation sets. The training sets consist of samples drawn with replacement, while the original dataset serves as the validation set.

Now let’s look at how to implement bootstrap sampling in python.
