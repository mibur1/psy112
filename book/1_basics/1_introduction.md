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

# <i class="fa-solid fa-code"></i> What is Machine Learning?
Have you ever asked yourself how Google Maps not only knows the way but also the fastest route? The secret behind it is called Machine Learning. By analyzing past traffic patterns, 
it predicts future conditions and adjusts routes in real time. What seems like magic is actually smart data processing which make predictions about the future given the past.


Machine learning is a field of artificial intelligence that allows computers to learn patterns from data and make predictions or decisions without being explicitly programmed. 
It relies on algorithms, which are step-by-step processes with a defined beginning and end, to analyze behavioral data, brain activity, and survey responses, 
helping to uncover meaningful patterns and predict outcomes.

## Types of learning

Machine learning can be categorized into three main types based on the way the model learns from data:

1. **Supervised Learning:**    The algorithm learns from labeled data, where each input has a corresponding output (e.g., predicting mental health scores based on questionnaire responses).
2. **Unsupervised Learning:**  The algorithm identifies patterns in unlabeled data, such as clustering individuals based on similar traits or behaviors.
3. **Reinforcement Learning:** The model learns by interacting with an environment and receiving feedback, akin to learning through trial and error (e.g., modeling decision-making processes in experiments).


## Classes of algorithms
In addition to the type of learning, algorithms can also be categorized based on the type of data that needs to be predicted:

| Type of data                  | Type of learning                                  | Type of learning                                       |
|-------------------------------|---------------------------------------------------|--------------------------------------------------------|
|                               | **Supervised learning** <br /> (labeled outcomes) | **Unsupervised learning** <br /> (no labeled outcomes) |
| **Qualitative (categorical)** | Classification                                    | Clustering                                             |
| **Quantitative (numerical)**  | Regression                                        | Dimensionality reduction <br /> Clustering             |


## Assessing performance

Assessing the quality of a machine learning model is crucial to ensure its reliability and applicability. Some important evaluation metrics are:

- **Mean Squared Error (MSE)**: Measures the average squared difference between predicted and actual values, particularly for regression tasks.
- **Accuracy**: Proportion of correctly predicted outcomes.

Additionally, techniques like **cross-validation** help estimate a model’s performance more reliably. Cross-validation splits the dataset into training and testing subsets, reducing the risk of overfitting and
proving a better assessment of how the model will perform on unseen data.

