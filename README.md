# Generative VS Discriminant
 - Generative algorithm: learning each structure, and then classifying it using the knowledge you just gained
 - Discriminant algorithm: determining the difference in the each without learning the structure, and then classifying the data_point.
<img src="https://user-images.githubusercontent.com/31917400/52206132-3a894180-2871-11e9-8cdd-81ac93c74e1d.jpg" />

A generative algorithm models how the data was "generated", so you ask it "what's the likelihood this or that class generated this instance?" and pick the one with the **better probability**. A discriminative algorithm uses the data to create a decision boundary, so you ask it "what side of the decision boundary is this instance on?" So it doesn't create a model of how the data was generated, it makes a model of what it thinks the boundary between classes looks like.

Since **discriminant** cares `P(Y|X)` only, while **generative** cares `P(X,Y) and P(X)` at the same time, in order to predict **P(Y|X)** well, the generative model has **less degree of freedom** in the model compared to discriminant model. So generative model is more robust, less prone to overfitting while discriminant is the other way around. So discriminative models usually tend to do better if you have lots of data; generative models may be better if you have some extra unlabeled data(the generative model has its own advantages such as the capability of dealing with missing data). 

## A> Discriminant Analysis
 - As a **Supervised method**, labels are used to learn the `data structure` which allows the **classification** of future observations.
 - Rule-based Classification Algorithm:
   - LDA(parametric), QDA
   - KNN
### 1. Linear Discriminant Analysis
# `P(g|x)`
 - Probability that `x` belongs to the group `g`. Which Grp they belong to? 
 - This probabilities come from a certain distribution(parametric)...in detail, 
 <img src="https://user-images.githubusercontent.com/31917400/52262491-86e18980-2924-11e9-9c4f-65a380b0c5c7.jpg" />

 - If assuming the Grp_feature distribution `P(x|g)` is **multivariate Gaussian**, what will happen?





## B> Generative Analysis
### 1. Latent Dirichlet Allocation
















































































