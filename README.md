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
 - Predict the membership of the given vector `x`. 
 - What's the probability that the vector `x` belongs to the Grp `g`? (p is the dimension of the vector x).
 - This probabilities come from a certain distribution of Grp(with different parametrization)...in detail, 
 <img src="https://user-images.githubusercontent.com/31917400/52262491-86e18980-2924-11e9-9c4f-65a380b0c5c7.jpg" />

 - So let's figure out the probability distribution `P(x|g)`. This is the distribution of data points in each group. 
 - If assuming the Grp_feature distribution `P(x|g)` is **multivariate Gaussian**, what will happen? (In the multivariate version, `µ` is a mean vector and σ is replaced by a covariance matrix `Σ`).
 <img src="https://user-images.githubusercontent.com/31917400/52270233-3d9b3500-2938-11e9-9585-63ef137328a4.jpg" />

# two functions to maximize `P(g|x)`.
 - So if we know the **distribution of x in each Grp: `P(x|g)`**, we can classify the new p-dimensional data points given in the future. We choose to have a Gaussian.  
 - Assumption: **all Grp share the equal `Σ` matrix**(in QDA, the equal covariance assumption does not hold, thus you cannot drop `-0.5log(|Σ|)` term).  
 - Which 'g' has the highest probability of owning the new p-dimensional datapoint? 
   - Eventually, Min/Max depends on the unique parameter(`µ,Σ`) of each Grp. 
   - When you plug in x vector, `µ,Σ` that minimizing **Mahalonobis Distance**, is telling you the membership of the vector `x`.  
   <img src="https://user-images.githubusercontent.com/31917400/52273637-57417a00-2942-11e9-8881-f7279ec947d4.jpg" />
   
   - When you plug in x vector, `µ,Σ` that maximizing **LD-function**, is telling you the membership of the vector `x`.
   <img src="https://user-images.githubusercontent.com/31917400/52273639-59a3d400-2942-11e9-900e-077ceabfb0b9.jpg" />
 
 - In practice,
 <img src="https://user-images.githubusercontent.com/31917400/52275546-c2da1600-2947-11e9-91fc-cdb49eec7c3e.jpg" />




### 2. K-Nearest Neighbors Analysis





## B> Generative Analysis
### 1. Latent Dirichlet Allocation
















































































