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
In the world of data science and machine learning, having access to large datasets is often a luxury rather than the norm. However, to build robust predictive models, we need effective ways to assess model performance and minimize the risk of overfitting. This is where resampling methods come into play.

Resampling methods involve **repeatedly drawing samples** from an available dataset to simulate independent datasets. These simulated sets help assess how well a trained model will perform on future data. While splitting data into training and test sets is a common practice, the results can vary depending on how the split is made. Resampling techniques provide a more reliable way to estimate model performance, reducing variability and improving accuracy.

The two mostly used resampling methods are:
- **Cross Validation** - creates non-overlapping substests that can be used to estimate the test error assocciated with a model
- **Bootstrapping** - samples with replacement, resulting in (partly) overlapping samples

```{admonition} Summary
:class: hint

Resampling methods involve:
1. Repeatedly drawing a sample from an existing dataset
2. fit the model to all resulting subsets and predict a held out amount of data
3. Examine all of the refitted models and draw appropriate conclusions
```

## **Cross-Validation**
### *Todays data - Iris dataset*
Let's look at how to apply the validation set approach using data.
The data set consists of 50 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor). Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters.

```{code-cell} 
# import packages
from sklearn import datasets
import matplotlib.pyplot as plt

# load dataset
iris= datasets.load_iris()

# Lets visualize two of our features to get an impression of the data
fig, ax = plt.subplots()
scatter = ax.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target)
ax.set(xlabel=iris.feature_names[0], ylabel=iris.feature_names[1])
fig= ax.legend(
    scatter.legend_elements()[0], iris.target_names, loc="lower right", title="Classes"
)
```
The goal of the algorithm is to classify the flowers based on our features. As we only have 150 datapoints for this prediction, we can use resampling methods to avoid overfitting and get a more stable result.

```{code-cell} ipython3
:tags: [remove-input]
from jupyterquiz import display_quiz
display_quiz('Quiz/Quiz_Iris.json')
```

### Validation Set approach

```{margin}
Hyperparameter are paramateres that are not learned from the data but set by the scientist before the training process begin. T
```

In the simplest approach of cross-validation the dataset is **randomly split into two independent subsets**:
- *Training Set*: Used to train the model by learning patterns and relationships in the data
- *Validation Set*: Used to assess model performance across different models and hyperparameter choices. It therefore provides an estimation of the test error.

```{code-cell} ipython3
:tags: [remove-input]
## creating a nice looking figure to visualize the spliiting in validation set approach
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Create figure and axis
fig, ax = plt.subplots(figsize=(8, 3), dpi=200)
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis("off")

# Whole dataset
ax.add_patch(patches.Rectangle((1, 4), 8, 1, color='gray', alpha=0.6))
ax.text(5, 4.5, "Whole Data Set", ha='center', va='center', fontsize=12, color='black')

# Validation set
ax.add_patch(patches.Rectangle((1, 2), 4, 1, color='lightcoral', alpha=0.6))
ax.text(3, 2.5, "Validation Set", ha='center', va='center', fontsize=12, color='black')

# Training set
ax.add_patch(patches.Rectangle((5, 2), 4, 1, color='lightblue', alpha=0.6))
ax.text(7, 2.5, "Training Set", ha='center', va='center', fontsize=12, color='black')

# Arrow from Whole Data Set to Training/Validation Set
ax.annotate('', xy=(3, 4), xytext=(3, 3), arrowprops=dict(arrowstyle='->', color='black'))
ax.annotate('', xy=(7, 4), xytext=(7, 3), arrowprops=dict(arrowstyle='->', color='black'))
# to Micha: Arrows even needed??

# Title
ax.text(5, 5.5, "Validation Set Approach", ha='center', va='center', fontsize=14, fontweight='bold')

plt.show()
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

# Using a support vector machine classifier with the training data
# C as hyperparameter/Regularization parameter
clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
```

4. **Evaluate** model performance
```{code-cell}
clf.score(X_test, y_test)

```
In the case of classification, we evaluate model performance using accuracy. An accuracy of ~97% means that, on average, the model correctly predicts 97 out of 100 test samples.


```{code-cell} ipython3
:tags: [remove-input]

from jupytercards import display_flashcards
display_flashcards('Quiz/Flashcard_ValidationSet.json')

```

```{admonition} Summary
:class: hint

While the Validation Set Approach is a quick and easy way to check how well a model performs, it has a major flaw: it puts all its trust in a **single data split**. Imagine training a model, evaluating it once, and calling it a day — what if that split was just lucky (or unlucky)? A bad split can doom a great model, while a lucky one might trick us into thinking a weak model is stronger than it is.
```

### k-fold Cross Validation
To get a more reliable and robust performance estimate, we need something smarter— something that doesn’t leave our results up to chance. Rather than worrying about which split of data to use for training versus validation, we'll use them all in turn.

In k-fold cross validation we randomly dive the dataset into k equal-sized folds. One fold is designated at the validation set, while the remaining **k-1** samples are the training sets. The fitting process is repeated **k-times**, each time using a different fold as the validation set. At the end of the process, we compute the **average** score across all validation sets to obtain a more reliable estimate of the model's overall performance.


```{code-cell} ipython3
:tags: [remove-input]

import matplotlib.pyplot as plt
import matplotlib.patches as patches
# documentation:
# https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Rectangle.html


# defining k 
k=5

# Create figure and axis
fig, ax = plt.subplots(figsize=(8, 3))
ax.set_xlim(0, 13)
ax.set_ylim(0, k+4)
ax.axis("off")

# Whole dataset
ax.add_patch(patches.Rectangle((1, 8), 10, 1, color='gray', alpha=0.6))
ax.text(6, 8.5, "Whole Data Set", ha='center', va='center', fontsize=12, color='black')

# k=1 
# Rectangle(xy, width, height...)
ax.add_patch(patches.Rectangle((1, 2), 2, 1, facecolor='lightcoral', alpha=0.6))        # Valdiation set
ax.add_patch(patches.Rectangle((3, 2), 8, 1, facecolor='lightblue', alpha=0.6))         # Training set 

# k=2 
ax.add_patch(patches.Rectangle((1, 3), 2, 1, facecolor='lightblue',  alpha=0.6))  # First training part
ax.add_patch(patches.Rectangle((3, 3), 2, 1, facecolor='lightcoral',  alpha=0.6))  # Validation part
ax.add_patch(patches.Rectangle((5, 3), 6, 1, facecolor='lightblue', alpha=0.6))   # Remaining training part

# k=3
ax.add_patch(patches.Rectangle((1, 4), 4, 1, facecolor='lightblue',  alpha=0.6)) 
ax.add_patch(patches.Rectangle((5, 4), 2, 1, facecolor='lightcoral',  alpha=0.6))   
ax.add_patch(patches.Rectangle((7, 4), 4, 1, facecolor='lightblue', alpha=0.6))  

# k=4
ax.add_patch(patches.Rectangle((1, 5), 6, 1, facecolor='lightblue',  alpha=0.6)) 
ax.add_patch(patches.Rectangle((7, 5), 2, 1, facecolor='lightcoral',  alpha=0.6))   
ax.add_patch(patches.Rectangle((9, 5), 2, 1, facecolor='lightblue', alpha=0.6))  

# k=5
ax.add_patch(patches.Rectangle((1, 6), 8, 1, facecolor='lightblue',  alpha=0.6)) 
ax.add_patch(patches.Rectangle((9, 6), 2, 1, facecolor='lightcoral',  alpha=0.6))   


# Arrow from Whole Data Set to Training/Validation Set
ax.annotate('', xy=(6, 7), xytext=(6, 8), arrowprops=dict(arrowstyle='<-', color='black'))

plt.show()
# probably not the most efficient way! Maybe putting it in a loop? 
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
