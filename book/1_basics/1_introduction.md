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

Machine learning is a field of artificial intelligence that allows computers to learn patterns from data and make predictions or decisions without being explicitly programmed. It has applications in psychology, such as predicting outcomes from behavioral data, analyzing brain data, or understanding patterns in survey responses.


## Types of learning

Machine learning can be categorized into three main types based on the way the model learns from data:

1. **Supervised Learning:**    The algorithm learns from labeled data, where each input has a corresponding output (e.g., predicting mental health scores based on questionnaire responses).
2. **Unsupervised Learning:**  The algorithm identifies patterns in unlabeled data, such as clustering individuals based on similar traits or behaviors.
3. **Reinforcement Learning:** The model learns by interacting with an environment and receiving feedback, akin to learning through trial and error (e.g., modeling decision-making processes in experiments).


## Classes of algorithms

| Type of data                  | Type of learning                                  | Type of learning                                       |
|-------------------------------|---------------------------------------------------|--------------------------------------------------------|
|                               | **Supervised learning** <br /> (labeled outcomes) | **Unsupervised learning** <br /> (no labeled outcomes) |
| **Qualitative (categorical)** | Classification                                    | Clustering                                             |
| **Quantitative (numerical)**  | Regression                                        | Dimensionality reduction <br /> Clustering             |


## Assessing performance

Assessing the quality of a machine learning model is crucial to ensure its reliability and applicability. Some important evaluation metrics are:

- Mean Squared Error (MSE): Measures the average squared difference between predicted and actual values, particularly for regression tasks.
- Accuracy: Proportion of correctly predicted outcomes.
- Cross-Validation: Divides the dataset into training and testing subsets to prevent overfitting.
