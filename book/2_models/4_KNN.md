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

# <i class="fa-solid fa-arrows-to-dot"></i> K-Nearest Neighbors

So far, we've explored regression problems where the outcome is numerical. But what if we want to predict **qualitative** outcomes—like a person’s mental health diagnosis based on their behavioral data? This is where **classification** methods come into play. 


One of the simplest yet effective classification algorithms is **k-Nearest Neighbors (kNN)**. The core idea is simple: observations that are similar tend to belong to the same class. In practice, this means that to classify a new data point, we look at the *k* closest points in the training data and assign the most common class label among them.

```{figure} figures/KNN.png
:alt: KNN
:width: 800
:align: center

source: https://www.ibm.com/think/topics/knn
```

While concepts like the **bias-variance trade-off** still apply, traditional regression metrics like Mean Squared Error aren't useful here. Instead, we often evaluate kNN classification using **error rate** —the proportion of misclassified observations.

In summary, kNN is a straightforward and powerful method for classification, relying on the simple assumption that similar inputs lead to similar outputs. Its ease of use and intuitive appeal make it a foundational technique in machine learning. Let's have a look on a practical application of kNN.

----------------------------------------------------------------
## *Todays data - Iris dataset*
As we’ve already worked with this dataset, it may look familiar. It contains measurements of three different iris species—Setosa, Versicolor, and Virginica—based on their sepal and petal lengths and widths.


```{code-cell}
from sklearn import datasets

# Load the iris dataset
iris = datasets.load_iris()
``` 
To refresh your memory, let’s visualize the dataset using the sepal length and sepal width features:

```{code-cell} ipython3
:tags: [remove-input]

import matplotlib.pyplot as plt
# Create a scatter plot for the first two features: sepal length and sepal width
_, ax = plt.subplots()
scatter = ax.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target)
ax.set(xlabel=iris.feature_names[0], ylabel=iris.feature_names[1])
_ = ax.legend(
    scatter.legend_elements()[0], iris.target_names, loc="lower right", title="Classes"
)
``` 
From the plot, you can already see that the Setosa species stands out clearly—it tends to have shorter and wider sepals, making it easily distinguishable. However, Versicolor and Virginica show more overlap, making them harder to separate using only these two dimensions.


The question is: Given a **new data point** with certain sepal measurements, how can we decide which iris species **it belongs to**? This is exactly the kind of problem that a classification algorithm like k-Nearest Neighbors is designed to solve. 


Let's first prepare our dataset:

```{code-cell}
import pandas as pd 

# Defining features and target
# Features: sepal length and width; target: type of flower

# Convert data (features) to a DataFrame
targets = pd.DataFrame(iris.data, columns=iris.feature_names)

X = targets[["sepal length (cm)", "sepal width (cm)"]]
y = iris.target
```
When training a kNN classifier, it's essential to **normalize the features**. This because kNN relies on distance calculations, and unscaled features can distort the results.

```{code-cell}
from sklearn.preprocessing import StandardScaler

# Scale the features using StandardScaler
scaler = StandardScaler()
# scale the entire feature set
X_scaled = scaler.fit_transform(X)

```

-----------------------------------------

## KNN Classifier Implementation

```{margin}
k is the number of nearest neighbors to use and is a hyperparameter
```
The choice of *k* plays a crucial role:
- A small *k* (e.g., 1) makes the method sensitive to noise and outliers.
- A larger *k* smooths the decision boundary, possibly at the cost of ignoring finer local structures.

####MICHA:  hier bitte INTERACTIVE PLOT - MIT FESTER DECISION BOUNDARY FINDEST DU EINEN PLOT WEITER UNTEN IM SCRIPT 


Unfortunately, there’s no magical formula to determine the best value for *k* in advance. Instead, we need to try out a range of values and use our best judgment to choose the one that works best.


To do this, we’ll fit the k-Nearest Neighbors model using different *k*-values within a specified range. To evaluate which value performs best, we use cross-validation — specifically, 5-fold cross-validation. Since cross-validation handles the splitting of the data into training and test sets internally, we don’t need to manually divide the dataset beforehand. 

1) Identifying the best *k*!
```{code-cell}
import numpy as np 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# range 1 to 45 in steps of one
k_range= list(range(1, 46))


# variable to store the accuracy scores in loop
scores= []

# loop trough the range of k using cross validation
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, X_scaled, y, cv=5)      # get scores for each k
    scores.append(np.mean(score))                 # append mean score to list

```


```{code-cell}
import matplotlib.pyplot as plt
import seaborn as sns

sns.lineplot(x = k_range, y = scores, marker = 'o')
plt.xlabel("K Values")
plt.ylabel("Accuracy Score")
```

```{code-cell}
best_index = np.argmax(scores)
# getting best k
best_k = k_range[best_index]
# getting accuracy of best k
best_score = scores[best_index]

print(f"Best k: {best_k}")
print(f"Accuracy with best k: {best_score:.4f}")
```

```{code-cell} ipython3
:tags: [remove-input]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Load iris data and select only 2 features for 2D plotting
iris = datasets.load_iris()
X = iris.data[:, :2]  # only sepal length and width
y = iris.target
feature_names = iris.feature_names[:2]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Use the best k from your cross-validation 
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_scaled, y)

# Create a meshgrid to evaluate the model
x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))
grid = np.c_[xx.ravel(), yy.ravel()]
Z = knn.predict(grid).reshape(xx.shape)

# Plot the decision boundary
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')

# Plot the training points
for i, label in enumerate(iris.target_names):
    plt.scatter(
        X_scaled[y == i, 0],
        X_scaled[y == i, 1],
        label=label,
        edgecolor='k'
    )

plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.title(f"Decision Boundary with k = {best_k}")
plt.legend(loc="lower right")
plt.show()

```

```{code-cell} ipython3
:tags: ["remove-input"]
from jupyterquiz import display_quiz
display_quiz("quiz/KNN.json", shuffle_answers=True)
```

```{admonition} The choice of k
:class: note 

In datasets where class boundaries are clear and the data is clean, a high k can actually be beneficial. It provides a form of regularization by averaging over many neighbors. In contrast, for more complex or noisy datasets, such a high k could oversimplify the structure and reduce model performance.
```

<br>
<br>

2) Train the model using the best *k*!
We can now train our model using the best *k* value using the code below. 

**Hands on:**


Now that we've determined the best value for k, we can go ahead and train and evaluate our final kNN model using this value. To properly assess how well the model performs on unseen data, we need to split our dataset into training and test sets. It's important to follow the correct sequence here: **first, we split the data — and only then do we scale it.** 


(TO MICHA: MUSS DAS ÜBERHAUPT NOCH SEIN? WIR HABEN OBEN JA 5 fold CV GENUTZT UND DAMIT JA EIG TEST UND TRAININGS DATA SCHON IN DIE BERECHNUNG DER ACCURACY MIT EINGEBUNDEN ODER?? https://www.datacamp.com/tutorial/k-nearest-neighbor-classification-scikit-learn HIER HABEN DIE DAS SO GEMACHT; DESWEGEN MACH ICH ES JETZT NOCH - ABER DU KANNST ES JA EINFACH LÖSCHEN WENN NICHT NÖTIG)

<iframe src="https://trinket.io/embed/python3/8c050c48e2b4" width="100%" height="356" frameborder="0" marginwidth="0" marginheight="0" allowfullscreen></iframe>


```{code-cell} ipython3
:tags: [remove-input]

# import packages
import pandas as pd
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


## PREPARE THE DATA
iris = datasets.load_iris()

# Convert data (features) to a DataFrame
targets = pd.DataFrame(iris.data, columns=iris.feature_names)

X = targets[["sepal length (cm)", "sepal width (cm)"]]
y = iris.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



## TRAIN THE MODEL USING THE BEST K!
# Train the model using training data
knn = KNeighborsClassifier(n_neighbors=31)
knn.fit(X_train, y_train)

#The model is now trained using the training data. Next, we can use it to make predictions on the test set. 

# predict the feature-category with the trained model
y_pred = knn.predict(X_test)

# check accuracy
accuracy = accuracy_score(y_test, y_pred)
```
