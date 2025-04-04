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
    hyperparam: 1
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

The goal of our model is to classify the flowering plants based on two features shown in the plot (sepal length and width). Which of the following is true about the model and task at hand?

```{code-cell} ipython3
:tags: [remove-input]
from jupyterquiz import display_quiz
display_quiz('quiz/iris.json')
```

## Validation Sets

```{margin}
Hyperparameters are parameters that are not learned from the data but set by the researcher before the training process.
```

The simplest form of cross validation is to simply split the dataset into two parts:

- *Training Set*: Part of the data used for trainin
- *Validation Set*: Part of the data used for testing (e.g. across different models and hyperparameters{{hyperparam}})


```{figure} figures/ValidationSet.drawio.png
:name: VS
:alt: Validation Set approach
:align: center

The validation set splits the dataset into a training and a testing set (these do not necessarily need to be of equal size).
```

The training and testing set neither need to be of equal size nor do they need to be contiguous blocks in the data. Let's try the validation set approach on the `Iris` data:

1.  Define features and target data

```{code-cell}
iris = datasets.load_iris(as_frame=True)

# Features: sepal length and width; target: type of flower
X = df[["sepal length (cm)", "sepal width (cm)"]] 
y = df["target"]
```

2. Split the data into training and test samples

```{code-cell}
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
```

3. Fit the model (we use a support vector classifier which you will learn about later in the seminar)

```{code-cell}
from sklearn import svm

model = svm.SVC(kernel='linear')
fit = model.fit(X_train, y_train)
```

4. Evaluate model performance

```{code-cell}
fit.score(X_test, y_test)
```

The `score()` method returns the accuray of our predictions. In this case, our algorithm correctly predicted the species of the flower in 85% of cases.


```{code-cell} ipython3
:tags: [remove-input]

from jupytercards import display_flashcards
display_flashcards('quiz/validation_set.json');
```

**Hands on**: In the editor below, perform classification for two splits in the the data. First, use 80% of the data for testing and 20% for training, and second use 20% for training and 80% for testing. Before evaluation the results, think about what kind of results you would expect from the two models. Which do you think will perform better?

<iframe src="https://trinket.io/embed/python3/48c2802e1e16" width="100%" height="356" frameborder="0" marginwidth="0" marginheight="0" allowfullscreen></iframe>

```{admonition} Summary
:class: hint

The validation set approach is a quick and easy way to check how well a model performs. However, it has a major flaw: it puts all its trust in a single data split which can doom a great model or trick us into thinking a weak model performs better than it actually does.
```

### K-fold Cross Validation (CV)

To get a more reliable and robust performance estimate, we need something smarter — something that doesn’t leave our results up to chance. Rather than worrying about which split of data to use for training versus validation, we'll use them all in turn.

In k-fold CV we randomly dive the dataset into $k$ equal-sized folds. One fold is designated at the validation set, while the remaining $k-1$ samples are the training sets. The fitting process is repeated $$k$-times, each time using a different fold as the validation set. At the end of the process, we compute the average score across all validation sets to obtain a more reliable estimate of the model's overall performance.


```{figure} figures/CV.drawio.png
:name: CV
:alt: Cross validation
:align: center

K-fold cross validation splits the dataset into $k$ equally sized parts and then trains the model on all posible combinations of it, keeping the proportion of train/test data constant.
```

Let`s try it on our data:


1. Defining *k*

```{code-cell} ipython3
from sklearn.model_selection import KFold, cross_val_score
k_fold = KFold(n_splits = 5)
```

2. Defining and evaluating the model

```{code-cell} ipython3
model = svm.SVC(kernel='linear')
scores = cross_val_score(model, X, y, cv=k_fold) 

print(f"Average accuracy:    {scores.mean()}")
print(f"Indidual accuracies: {scores}")
```

If we are interested in the exact models, we can also run the training and evaluation explicitly which allows us to save the models:

```{code-cell} ipython3
model = svm.SVC(kernel='linear')
score_list = []
model_list =  []

for train_index, test_index in k_fold.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index] # iloc because X is a df
    y_train, y_test = y.iloc[train_index], y.iloc[test_index] # iloc because y is a df

    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    
    score_list.append(score)
    model_list.append(model)

print(f"Best performing model in split {score_list.index(max(score_list))}.")
```

```{admonition} Validatation set vs. k-fold
:class: note

Comparing the two approaches, we see that the validation set approach shows a higher accuracy compared to CV. This tells us that our initial estimates were probably overly optimistic.
```

**Try it yourself**: Change the number of folds $k$ and observe how the predicitions change. What do you feel like is a good tradeoff between bias and variance?

<iframe src="https://trinket.io/embed/python3/679ce3200f38" width="100%" height="356" frameborder="0" marginwidth="0" marginheight="0" allowfullscreen></iframe>

```{admonition} The choice of $k$
:class: note 

Choosing an appropriate k involves a tradeoff between bias, variance, and computational cost. A higher k generally provides a more stable and reliable estimate but comes with higher computational cost and also requires a sufficiently big dataset to still have a representative test set.
 
Generally speaking, $k=5$ or $k=10$ are common choices.
```


### Leave-one-out Cross Validation (LOOCV)

LOOCV is a special case of k-fold cross validation, where $k$ equals the number of observations. In LOOCV, the model is trained on all but one data point, and the remaining single observation is used for validation. This process repeats for each data point, ensuring every observation is used for testing exactly once. 

While LOOCV provides a low-bias estimate, it is computationally expensive and may lead to high variance in model performance. The implementation is fairly similar, we just need to change the CV from `KFold()` to `LeaveOneOut()`:

```{code-cell} ipython3
from sklearn.model_selection import LeaveOneOut

model = svm.SVC(kernel='linear')
loocv = LeaveOneOut()

scores = cross_val_score(model, X, y, cv = loocv)

print(f"Average accuracy:    {scores.mean()}")
print(f"Indidual accuracies: {scores}")
```


## Bootstrapping

Bootstrapping is a resampling technique in statistics and machine learning that repeatedly draws samples *with replacement* from a dataset to estimate a population parameter. It can be used to quantify the uncertainty associated with a given estimator or statistical learning method.  

As in CV, bootstrapping also uses training and validation sets. The training sets consist of samples drawn with replacement, while the original dataset serves as the validation set:
