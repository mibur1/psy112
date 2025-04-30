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

# <i class="fa-solid fa-brain"></i> Neural Networks

https://mlu-explain.github.io/neural-networks/

This notebook walks through a **single hidden‐layer neural network** (“vanilla” perceptron) from theory to practice, using Python’s scikit-learn.  We’ll cover:

1. **What is a neural network?**  History, motivation, and terminology.  
2. **Network architecture**: inputs, hidden layer, activation, outputs.  
3. **Training**: cost functions, backpropagation, regularization, early stopping.  
4. **Hands-on examples**:  
   - Regression: approximating a noisy sine curve  
   - Classification: the Iris dataset  
5. **Hyperparameter exploration**: hidden‐unit counts, penalty strength.  
6. **Key best practices** and next steps.

---

## 1. What Is a Neural Network?

A neural network (NN) is a flexible **nonlinear statistical model** inspired by the brain’s structure.  In machine learning, a NN is:

- **Layered**: an input layer, one or more hidden layers, and an output layer.  
- **Parametric**: connections (edges) have weights and each node has a bias.  
- **Nonlinear**: hidden units apply an activation function (sigmoid, ReLU, etc.) to weighted sums.  

> **“Vanilla” single‐hidden‐layer network**  
> - One hidden layer only  
> - Often called a backpropagation network or single‐layer perceptron  
> - Capable of approximating any continuous function given enough hidden units (universal approximation theorem).

### Key components

1. **Inputs** \(X\): raw features or predictors.  
2. **Weights** \(W\) and **biases** \(b\): parameters learned from data.  
3. **Hidden layer**: derived features  
   \[
     z^{(1)} = W^{(1)} X + b^{(1)},\quad
     a^{(1)} = \sigma(z^{(1)})
   \]  
4. **Output layer**: produces prediction  
   - **Regression**: single linear unit  
   - **Classification**: one unit per class + softmax for probabilities.  

---

## 2. Setup: Imports and Environment

We’ll use:
- `numpy` for arrays  
- `matplotlib` for plots  
- `scikit-learn` for MLPRegressor/Classifier, train/test split, scaling  

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
%matplotlib inline
```

## 3. Regression Example: Approximating a Noisy Sine Curve

**Goal:** Fit \(y = \sin(x)\) over \([0, 2\pi]\) with additive noise.  Demonstrates how a simple NN can learn a smooth nonlinear function.

1. **Data generation**  
2. **Train/test split**  
3. **Feature scaling** is critical for stable training  
4. **Train** an MLPRegressor with one hidden layer of 50 units  
5. **Plot** true vs. predicted  

```{code-cell} ipython3
# 1. Generate data
X = np.linspace(0, 2 * np.pi, 200).reshape(-1, 1)
y = np.sin(X).ravel() + 0.1 * np.random.randn(200)

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# 3. Scale
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# 4. Train
model = MLPRegressor(hidden_layer_sizes=(50,),
                     activation='relu',
                     solver='adam',
                     max_iter=1000,
                     random_state=42)
model.fit(X_train_s, y_train)

# 5. Evaluate & plot
X_plot = np.linspace(0, 2*np.pi, 500).reshape(-1,1)
X_plot_s = scaler.transform(X_plot)
y_plot = model.predict(X_plot_s)

plt.figure(figsize=(8,4))
plt.scatter(X_train, y_train, label='Train', alpha=0.6)
plt.scatter(X_test,  y_test,  label='Test',  alpha=0.6)
plt.plot(X_plot, np.sin(X_plot), linestyle='--', label='True sine')
plt.plot(X_plot, y_plot,            label='NN fit')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Neural Net Approximation of sin(x)')
plt.legend()
plt.show()
print(f"Test MSE: {np.mean((model.predict(X_test_s)-y_test)**2):.4f}")
```

### Discussion

- The **mean squared error (MSE)** on the test set shows generalization quality.  
- If you see **wiggles**, you may have too many hidden units or too little regularization.  
- **Always scale** your inputs; otherwise training can fail or converge extremely slowly.

---

## 4. Classification Example: Iris Dataset

**Goal:** Classify Iris species (3 classes) from 4 measurements.  Illustrates multiclass output and softmax.

1. **Load data** from scikit-learn  
2. **Split & scale**  
3. **Train** MLPClassifier with 50 hidden units  
4. **Report** accuracy  

```{code-cell} ipython3
# 1. Load data
iris = load_iris()
X, y = iris.data, iris.target

# 2. Split & scale
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# 3. Train classifier
clf = MLPClassifier(hidden_layer_sizes=(50,),
                    activation='relu',
                    solver='adam',
                    max_iter=1000,
                    random_state=42)
clf.fit(X_train_s, y_train)

# 4. Evaluate
acc = clf.score(X_test_s, y_test)
print(f"Iris classification accuracy: {acc*100:.1f}%")
```

### Learning Curve and Convergence

You can inspect `clf.loss_curve_` to see how the training loss decreases over iterations, and use `early_stopping=True` to avoid overfitting.

```{code-cell} ipython3
plt.figure()
plt.plot(clf.loss_curve_)
plt.xlabel('Iteration')
plt.ylabel('Training Loss')
plt.title('Loss Curve for Iris Classifier')
plt.show()
```

## 5. Exploring Hyperparameters

**What happens when we change hidden‐unit count?**  
Too few units → underfit (high bias).  
Too many units → overfit (high variance) unless regularized.

```{code-cell} ipython3
hidden_sizes = [5, 20, 50, 100]
train_losses = []
for h in hidden_sizes:
    m = MLPRegressor(hidden_layer_sizes=(h,),
                     max_iter=1000, random_state=42)
    m.fit(X_train_s, y_train)
    train_losses.append(m.loss_)

plt.figure()
plt.plot(hidden_sizes, train_losses, marker='o')
plt.xlabel('Number of Hidden Units')
plt.ylabel('Final Training Loss')
plt.title('Effect of Hidden Layer Size')
plt.show()
```

## 6. Regularization & Early Stopping

Neural nets easily overfit.  Two common remedies:

1. **Weight decay** (L2 penalty via `alpha` parameter).  
2. **Early stopping**: monitor validation set loss and stop when it rises.

```{code-cell} ipython3
model_reg = MLPRegressor(hidden_layer_sizes=(50,),
                         alpha=0.01,
                         early_stopping=True,
                         validation_fraction=0.2,
                         n_iter_no_change=10,
                         max_iter=1000,
                         random_state=42)
model_reg.fit(X_train_s, y_train)
print("Test R²:", model_reg.score(X_test_s, y_test))
print("Number of iterations:", model_reg.n_iter_)
```

## 7. Summary & Next Steps

- **Neural networks** are powerful for capturing complex nonlinear patterns.  
- **Key knobs**:  
  - Hidden layer size(s)  
  - Activation function  
  - Regularization strength (`alpha`)  
  - Learning rate / solver choices  
- **Best practices**:  
  - Always scale inputs.  
  - Use validation or cross-validation to tune hyperparameters.  
  - Monitor **loss curves** for convergence and overfitting.  
- **Extensions**:  
  - Deeper networks with multiple hidden layers.  
  - Convolutional architectures for images.  
  - Recurrent networks for sequences.

>*Exercises to try:* vary activation functions, compare solvers (`sgd` vs. `adam`), add dropout with a more flexible framework (e.g. Keras/TensorFlow).