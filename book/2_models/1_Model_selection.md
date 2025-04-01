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

# <i class="fa-brands fa-python"></i> Model Selection
In neurocognitive psychology, brain imaging (e.g., fMRI, EEG), cognitive assessments, and behavioral experiments generate vast datasets, measuring thousands of brain regions, connectivity patterns, and behavioral traits. This data captures everything from patient vitals to cognitive processes and offers **detailed insights** with immense predictive power. However, it also introduces challenges: **too many predictors**, making analysis and interpretation difficult.

## The large p issue
**Big Data** refers to large datasets with many predictors, that cannot be processed or analyzed using traditional data processing techniques. For our prediction models, this brings some issues:
- While the linear model can in theory still be used for such data, the **ordinary least squares fit becomes infeasible**, especially when p > n 
- The large amount of features reduce interpretability


This is where **linear model selection** becomes essential, offering techniques to refine our models and extract meaningful insights from high-dimensional neurocognitive data!

----------------------------------------------------------------
### *Todays data with many predictors - Hitters dataset*
For pracitcal demonstration, we will use the `Hitters` dataset. This data set provides Major League Baseball Data from the 1986 and 1987 seasons. It contains 322 observations of major league players on 20 variables. The Research aim is to predict a baseball player's salary on the basis of various predictors associated with the performance in the previous year.

```{code-cell} 
# import packages
import statsmodels.api as sm 

# get dataset
hitters = sm.datasets.get_rdataset("Hitters", "ISLR").data
```
Get yourself familiar with the dataset. Look at the predictor variables. Which information do we include to predict the salary? 
You can check the variable names here: https://islp.readthedocs.io/en/latest/datasets/Hitters.html  
Also take a closer look to the variable you want to predict! Do we have the information(s) that we need for all players?

<iframe src="https://trinket.io/embed/python3/12222a549d51" width="100%" height="356" frameborder="0" marginwidth="0" marginheight="0" allowfullscreen></iframe>

For computationally reasons, we will not include all predictors but only a small subset
```{code-cell} 
# keeping a total of 10 variables - the target ´Salary´ and 9  features.
hitters_subset = hitters[["Salary", "CHits", "CAtBat", "CRuns", "CWalks", "Assists", "Hits", "HmRun", "Years", "Errors"]].copy()

# make sure the rows, containing missing values, are dropped
hitters_subset.dropna(inplace=True)

hitters_subset.head()
```

Let’s also take a look at the correlation between predictors to check for potential multicollinearity, which can affect the stability of linear regression models.

```{code-cell} 
import matplotlib.pyplot as plt
import seaborn as sns

# Correlation heatmap for hitters subset
plt.figure(figsize=(8, 5))
sns.heatmap(hitters_subset.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Hitters Subset", fontsize=16)
plt.show()
```
The heatmap reveals strong correlations between several predictors, indicating multicollinearity. Subset Selection methods are sensitive to multicollinearity and can make unstable or misleading variable selections.

- `CHits` and `CAtBat` show a correlation of 1,
- `CHits` and `CRuns` have a very strong correlation of 0.98,

In such cases, it may be beneficial to remove one of the correlated features.


**Hands on:** Again take a look at what the feautures are measuring (https://islp.readthedocs.io/en/latest/datasets/Hitters.html) and decide, which one to drop! To make your decision not just random, maybe think about the relevance in predicting the `Salary`. Drop the features and check the heatmap again!

<iframe src="https://trinket.io/embed/python3/643a5691f205" width="100%" height="356" frameborder="0" marginwidth="0" marginheight="0" allowfullscreen></iframe>

```{code-cell} ipython3
:tags: [remove-input]
## Removing features
# List of features to drop
features_drop = ["CRuns", "CAtBat"]
# dropping 
hitters_subset2= hitters_subset.drop(columns= features_drop)
```

Okay, now that we know our dataset, let's look at how to handle such a large number of predictors! 

----------------------------------------------------------------
## Handling big data in linear models


```{admonition} Handling big data
:class: hint

To handle large datasets efficiently in linear modeling, three key techniques are used:

- **Subset Selection**
- **Dimension Reduction**
- **Regularization / Shrinkage**
```

By leveraging these methods, we can build robust predictive models that remain efficient and interpretable, even in the face of Big Data challenges.


### Subset Selection
In subset selection we identify a subset of *p* predictos that are truly related to the outcome. The model get fitted using least squares on the reduces set of variables.

How do we determine which variables are relevant?! 

####  Best Subset Selection

```{image} ./figures/BestSubsetSelection.drawio.png
:alt: ModelSelection
:width: 30%
:align: left
```

```{margin}
The Null Model only predicts the sample mean
```
<br>
<br>

1. Consider all possible models
    - Starting with Null Model <em>M0</em>, which contains no predictors
    - Iteratively adding a predictor to the model
2. Identify the Best Model of each size
    - Either by the smallest RSS or the largest <code>R²</code></li>
3. Identify the Best Overall Model
    - Use cross-validation to find the best <em>Mk</em></li>

<br>
<br>
<br>

Let's get back to our dataset and see how Best Subset Seletion is performed in python.

```{code-cell} ipython3
:tags: ["remove-input"]
from jupyterquiz import display_quiz
display_quiz("quiz/BestSubsetSelection.json", shuffle_answers=False)
```

<br>

--------------------------------------
**To MICHA:** We could also think about using abess. https://github.com/abess-team/abess
But for now I decided to not use it since it just do everything and I thought it might be harder to understand the concept behind it, because we  . However if you decide to go that way, we can use the following code chunk.
We could also consider using abess (https://github.com/abess-team/abess) — it's a nice OPEN SOURCE library that performs best subset selection super efficiently.
However, I decided not to use it for now because it does everything automatically, and I thought that might make it harder for students to understand what's actually happening. but if you prefer to go use abess, here is the code chunk, we could use:
If you need more details:https://abess.readthedocs.io/en/latest/auto_gallery/1-glm/plot_1_LinearRegression.html#sphx-glr-auto-gallery-1-glm-plot-1-linearregression-py

```{code-cell} 
from abess.linear import LinearRegression 
import numpy as np
import pandas as pd

# Prepare data - defining target and features
X= np.array(hitters_subset.drop("Salary", axis=1))
y= np.array(hitters_subset["Salary"])

# Use abess for best subset selection
model = LinearRegression(support_size=range(1, 10))  # support_size is how many features to try
model.fit(X, y)

# Get selected features (non-zero coefficients)
ind = np.nonzero(model.coef_)
print("non-zero:\n", hitters_subset.columns[ind])
print("coef:\n", model.coef_)
```
The abess algorithm evaluates all possible combinations of our 9 predictors and automatically selects the best subset based on internal criteria (e.g., minimizing BIC). In our case, it selected just two predictors: `CAtBat` and `Assists` 

You can also implement Best Subset Selection manually using the `mlxtend` library. Unlike abess, this approach allows you to explicitly control and understand what’s happening at each step of the selection process.
------------------------------- delete until here if we use the manual part!

In the following example, we evaluate all possible feature combinations from 1 to 9 predictors and identify the best subset.

Before even starting the subset selection process, the first step is to define the target and the features.

```{code-cell} 
# Define target and features
X = hitters_subset2.drop(columns=["Salary"])
y = hitters_subset2["Salary"]


```
Before performing the best subset selection, we split our data into training and test dataset. Although the selection function uses cross-validation to identify the best subset of predictors (Step 3), this evaluation is done during the selection process and can still overfit to the data. To fairly assess how the final model performs on new data, we split off a test set and use it only after feature selection is complete.

|Purpose                        	   | What is it for?                                   | When?                                               |
|------------------------------------- |---------------------------------------------------|-----------------------------------------------------|
|Cross Validation in selection function|Helps choose the best subset of features           |During selection                                     |
|Test Set Evaluation                   |Checks how well the final model performs           |After selection                                      |

```{code-cell} 
# Split data BEFORE selection
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
So far, so good — time to run best subset selection on the training data.

```{code-cell} 
:tags: [remove-output]

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from mlxtend.feature_selection import ExhaustiveFeatureSelector

# Preperation 
# Define the regression model 
model = LinearRegression()

# Use 5-fold cross-validation to evaluate model performance
cv_folds = 5

# Perform best subset selection 
efs = ExhaustiveFeatureSelector(model, 
          min_features=1, 
          max_features=7,             # Try all subsets from 1 to 7
          scoring='r2',               # R² as scoring metric
          cv=cv_folds,                # Apply k-fold cross-validation (e.g., 5-fold)
          print_progress=False)

# fit it only on trainings data!
efs.fit(X_train, y_train)
```
Let's have a look how the best model for each model size performed. This plot helps us visualize how performance improves as we increase the number of features. It reflects Step 2 of our selection process and gives us insight into the bias-variance tradeoff: at some point, adding more features doesn't necessarily improve the model much.

```{code-cell}
import matplotlib.pyplot as plt

# Extract metric results
metric_dict = efs.get_metric_dict()
results = []

for subset in metric_dict.values():
    n_features = len(subset['feature_idx'])
    avg_r2 = subset['avg_score']  
    results.append((n_features, avg_r2))

# Create DataFrame
results_df = pd.DataFrame(results, columns=["n_features", "avg_r2"])

# Aggregate by number of features
results_df = results_df.groupby("n_features", as_index=False).max()  # max R², not min

# Plot
plt.figure(figsize=(8, 5))
plt.plot(results_df["n_features"], results_df["avg_r2"], marker="o")
plt.title("Best Subset Selection: R² vs. Number of Features")
plt.xlabel("Number of Features")
plt.ylabel("Cross-validated R²")
plt.show()
```

Now we need to retrieve the best overall subset. Since efs performs cross-validation internally, we can simply identify the subset with the highest average R² score from the output.

```{code-cell}
# Getting the names of best features
print(f"Cross-validated R² of best model: {efs.best_score_:.4f}")
print("Best feature subset:", efs.best_feature_names_)
```
According to best subset selection, the subset with 4 predictors is the best combination, achieving a training R² score of 0.41.

#### Forward Stepwise Selection
Best subset selection is not feasible for very large *p* due to its computational demands. A more efficient way solving this problem, is foward stepwise selection. 

```{image} ./figures/ForwardStepwiseSelection.drawio.png
:alt: ModelSelection
:width: 30%
:align: left
```
<br>
<br>

1. Beginning with null hypothesis
2. Adding the most significant variables one after the other
    - Either by the smallest RSS or the largest <code>R²</code></li>
3. Repeat it until...
    - reaching a stopping criteria
    - k=p
4. Identifying the single best model using cross-validation

<br>
<br>
<br>

So let`s apply forward stepwise selection. 

After defining the predictors, response variable, and the number of cross-validation folds, we can now run the forward stepwise selection. Unlike exhaustive selection, which fits all possible models, forward selection starts with no predictors and adds one feature at a time — always choosing the one that improves performance the most. This process continues until we reach the maximum number of features (in this case, 9).If you'd rather stop after a specific number of features, you can control that using the `k_features` parameter.

```{code-cell}
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from sklearn.model_selection  import train_test_split

# Run forward selection using MSE as the scoring metric
forward = SequentialFeatureSelector(
    model,              # defined model 
    k_features=(1,7),   # Stopping criteria: Try models with 1,2,..,7 features
    forward=True,       # use forward selection 
    floating=False,     # Do not use floating selection -> Classic forward selection
                        # floating =True : after each addition, it also checks if it should remove a feature that has become less useful
    scoring='r2',       
    cv=cv_folds)

# Fit the prepared model on our trainings data
sfs = forward.fit(X_train, y_train)
```
Just like we did with best subset selection, we now visualize the R² score at each model size. This helps us understand how the model performance improves as we add more predictors.

```{code-cell} ipython3
:tags: [remove-input]
# Plot the R² score for each number of features
import matplotlib.pyplot as plt
import pandas as pd

# Access R² scores for each step (stored during selection)
r2_scores = []
num_features = []

# Extract metric info
metric_dict = sfs.get_metric_dict()

for k in metric_dict.keys():
    r2 = metric_dict[k]['avg_score']  # R² is already positive
    r2_scores.append(r2)
    num_features.append(len(metric_dict[k]['feature_idx']))

# Plot
plt.figure(figsize=(8, 5))
plt.plot(num_features, r2_scores, marker='o')
plt.title("Forward Selection: R² vs. Number of Features")
plt.xlabel("Number of Features")
plt.ylabel("Cross-validated R²")
plt.tight_layout()
plt.show()
```
This looks quite similar to best subset selection. Let’s now take a look at which predictors were selected by the forward stepwise selection model. We’ll extract the final subset of features that the algorithm identified as most predictive of `Salary`. 

```{code-cell}
# Get selected features
selected_features = list(sfs.k_feature_names_)
print("Selected features:", selected_features)

# Get cross-validated R² score of the best model
best_cv_r2 = sfs.k_score_  # no flipping needed!
print(f"Cross-validated R² of best model: {best_cv_r2:.4f}")
```

<br>

```{admonition} Interim Summary
:class: tip

- As we can see, we ended up with the same three predictors as in best subset selection - `CWalks`, `Hits`, `Errors`, `HmRun`. Once again, a combination of these three seems to perform best in predicting `Salary`. 
- However, this is not necessarily always the case — best subset and stepwise selection can, and often do, **result in different predictors** or even a different number of predictors being selected.
- In our case, we only had a small number of predictors, which makes it more likely to end up with the same subset.
```

To wrap up our subset selection methods, let’s briefly explore backward stepwise selection.


<iframe src="https://trinket.io/embed/python3/a00b2117cbe7" width="100%" height="356" frameborder="0" marginwidth="0" marginheight="0" allowfullscreen></iframe>

#### Backward Stepwise Selection
```{image} ./figures/BackwardStepwiseSelection.drawio.png
:alt: ModelSelection
:width: 30%
:align: left 
```

```{margin}
The Full Model contain all p predictors!
```

<br>
<br>

1. Beginning with the Full Model <em>Mp</em>
2. Iteratibely removes the least usefull predictor
3. Repeat it until...
    - reaching a stopping criteria
    - <em>k=0</em>
4. Identify the best overall model using cross-validation


<br>
<br>

Backward stepwise selection works very similarly to forward selection — the main difference is that we start with the full model and remove features one by one. 


```{code-cell} ipython3
:tags: [remove-input]

from jupytercards import display_flashcards
display_flashcards('quiz/BackwardSelection_Flashcard.json')
```


```{admonition} Subset Selection Summary
:class: tip

| Best Subset Selection            	  | Forward Stepwise Selection                        | Backward Stepwise Selection                         |
|-------------------------------------|---------------------------------------------------|-----------------------------------------------------|
|**-** computationally very expensive |**-** not guaranteed to find best model            |**-** not guaranteed to find best model              |
|**-** with many *p* may overfit      |**+** possible to us when p is very large          |**+** possible to us when p is very large, given p<n |
|**+** able to find the best model    |**+** computationally less demanding               |**+** computationally less demanding                 |
```


```{code-cell} ipython3
:tags: [remove-input]
from jupyterquiz import display_quiz
display_quiz('quiz/SubsetSelection.json')
```

#### What next?
Once we have identified the features that are relevant for predicting the outcome, let`s evaluate the model performance and estimate true test error with the 4 predictors identified by Best Subset Selection and Forward Stepwise Seletion.
```{code-cell}
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.linear_model import LinearRegression

selected_features = ['CWalks', 'Hits', 'HmRun', 'Errors']

# use splitted data only from selected features
X_train_sel = X_train[selected_features]
X_test_sel = X_test[selected_features]

# fit model with selected featues
model = LinearRegression()
model.fit(X_train_sel, y_train)

# Predict on the test set
y_pred = model.predict(X_test_sel)

# Evaluate
mse_test = mean_squared_error(y_test, y_pred)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(y_test, y_pred)

# Print results
print(f"Test MSE: {mse_test:.2f}")
print(f"Test RMSE: {rmse_test:.2f}")
print(f"Test R²: {r2_test:.4f}")
```
What does this mean for our prediction using the 4 features?
- On average, our predictions deviate from the actual salary by about $417.
- Our model explains only a small portion (~3%) of the variability in salary. That also means that the majority of factors influencing salary are not captured by these predictors.

### Dimension Reduction
Dimensionality reduction is a model selection technique that simplifies high-dimensional datasets by transforming them into a smaller set of uncorrelated components. Instead of selecting individual features, it **combines correlated variables into new components** that retain most of the data’s variance. This reduces computational cost, lowers the risk of overfitting, and improves model performance — especially when many features are present. The first component captures the most variance, the second the next most, and so on. This method helps preserve patterns and trends in the data while working with fewer, more manageable inputs.

```{admonition} TO MICHA - HEEEEELP
:class: danger

- 0.03 Test R^2? Das ist turbo schlecht!
- genau gleiche trainings R^2 für Forward and best subset slection? 
``` 