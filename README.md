### [Note] Variations on supervised and unsupervised Model ?
 - Semi-Supervised Learning:
   - we have a bunch of pairs (**x**1,**y**1), (**x**2,**y**2), ...(**x**_i,**y**_i), and then we are additionally given more x values such as x_i+1, x_i+2,..up to x_n. Our task is to predict `y_i+1`, `y_i+2`,..up to `y_n`.
 - Reinforcement Learning:
   - Investigate the "reward/loss"(long/short term payoff) associated with a certain action or state..
### [Note] PGM ?
If you are keen on studying generative models and delving deeper into them, I would say concepts and thorough knowledge on Probabilistic graphical models is essential. If your focus is on Discriminative models or planning to use deep learning as a blackbox then you can get away without PGM and its probably not very essential. But if you are planning for a research either in implicit or explicit generative models or especially deep generative models, then I strongly recommend PGM as a course. Its a valuable tool for sure

# Generative VS Discriminative Model
Machine Learning models can be typically divided into two types. Discriminative and Generative models. Discriminative models deal with classification or categorization of the data, examples include SVMs, linear/logistic regression, CRFs etc. Generative approaches model how the data is generated and various other tasks can be further addressed with this generative information, examples include HMMs, GMM, Boltzman machines. Deep learning models such as DBMs, Variational AutoEncoders are again generative models.
 - __[A]. Generative algorithm:__ learning each structure, and then classifying it using the knowledge you just gained.
   - A generative model is a statistical model of the joint probability distribution on `P(X,Y)` and Classifiers are computed using probablistic models.
   - Generative modeling means building a model that can generate new examples that come from the same distribution as the training data (or that can look at an input example and report the likelihood of it being generated by that distribution). This means generative modeling is a kind of unsupervised learning.
   - A generative algorithm models uses the data to create a **`probabilities`**, and how the data was "generated", so you ask it "what's the likelihood this or that class generated this instance?" and pick the one with the **better probability**. 
     - Estimate joint probability ### P(Y, X) = P(Y|X)f(X) = f(X|Y)P(Y) 
       - where Y is label(class), `f() is pdf` and `P() is class marginal probability`.
       - `f(X|Y)P(Y)` : first choose a class, then given the class, we choose(generate) the point X. 
       - P(Y|X)f(X) : first choose the point X, then given the point, we choose a class. This is discriminative though. 
     - Estimates not only probability of labels but also the features
     - Once model is fit, can be used to generate data, but often works worse, particularly when assumptions are violated
   - Linear Generative Dimensionality Reduction Algorithm
     - LDA, QDA, PCA, Naive Bayes, etc.
   - Nonlinear Generative Dimensionality Reduction Algorithm
     - AutoEncoder, Variational AutoEncoder, etc.
   
 - __[B]. Discriminative algorithm:__ determining the difference in the each without learning the structure, and then classifying the data_point.
   - A discriminative model is a statistical model of the conditional probability distribution on `P(Y|X=x)` and Classifiers computed **without using a probability model** are also referred to loosely as "discriminative".
   - A discriminative algorithm uses the data to create a **`decision boundary`**, so you ask it "what side of the decision boundary is this instance on?" So it doesn't create a model of how the data was generated, it makes a model of what it thinks the boundary between classes looks like.

Since **discriminative** cares `P(Y|X)` only, while **generative** cares `P(X,Y) and P(X)` at the same time, in order to predict **P(Y|X)** well, the generative model has **`less degree of freedom`** in the model compared to discriminative model. So generative model is more robust, less prone to overfitting while discriminative is the other way around. So **discriminative models** usually tend to do better if you have `lots of data`; **generative models** may be better if you have some extra `unlabeled or missing data`(the generative model has its own advantages such as the capability of dealing with missing data). 
<img src="https://user-images.githubusercontent.com/31917400/52206132-3a894180-2871-11e9-8cdd-81ac93c74e1d.jpg" />










## A> Generative Analysis

---------------------------------------------------------------------------------------------------------------------
## 1. Linear Discriminant Analysis
 - As a **Supervised method**, labels are used to learn the `data structure` which allows the **classification** of future observations.

# `P(g|x)`
 - Predict the membership of the given vector `x`. 
 - We have a dataset containing lots of vector observations(rows) and their labels. 
 - What's the probability that the new vector observation `x` belongs to the Grp `g`? (p is the dimension of the vector x).
 - This probabilities come from a certain **likelihood distribution of Grp**(with different parametrization)...in detail, 
 <img src="https://user-images.githubusercontent.com/31917400/52278515-2156c280-294f-11e9-9bc2-6e40c4563b8f.jpg" />

 - So let's figure out the Likelihood distribution `P(x|g)`. This is the distribution of data points in each group. If we know the **distribution of x in each Grp: `P(x|g)`**, we can classify the new p-dimensional data points given in the future...so done and dusted. What if choosing the Grp_feature distribution `P(x|g)` as **multivariate Gaussian** ? (Of course, in the multivariate version, `µ` is a mean vector and σ is replaced by a covariance matrix `Σ`).
 <img src="https://user-images.githubusercontent.com/31917400/52270233-3d9b3500-2938-11e9-9585-63ef137328a4.jpg" />

# two functions to maximize `P(g|x)`.
 - Assumption: **all Grp share the equal `Σ` matrix**(in QDA, the equal covariance assumption does not hold, thus you cannot drop `-0.5log(|Σ|)` term).  
 - Which 'g' has the highest probability of owning the new p-dimensional datapoint? 
   - Eventually, Min/Max depends on the unique parameter(`µ,Σ`) of each Grp. 
   - When you plug in x vector, `µ,Σ` that minimizing **Mahalonobis Distance**, is telling you the membership of the vector `x`.  
   <img src="https://user-images.githubusercontent.com/31917400/52273637-57417a00-2942-11e9-8881-f7279ec947d4.jpg" />
   
   - When you plug in x vector, `µ,Σ` that maximizing **LD-function**, is telling you the membership of the vector `x`.
   <img src="https://user-images.githubusercontent.com/31917400/52273639-59a3d400-2942-11e9-900e-077ceabfb0b9.jpg" />
 
 > In practice,
 <img src="https://user-images.githubusercontent.com/31917400/52275738-4dbb1080-2948-11e9-9768-3da4a0c5c773.jpg" />

 > **Log Odd Ratio** and `Linear Decision Boundary`
   - What if the Grp membership probability of 'g1', 'g2' are the same? 
   - Then we can say that the given vector point is the part of `Linear Decision Boundary` !!!
   <img src="https://user-images.githubusercontent.com/31917400/52283578-a2678700-295a-11e9-98ae-817a9f91afdc.jpg" />

 > LDA and Logistic regression
   - LDA is Generative while LogisticRegression is discriminative.
   - LDA operates by maximizing the log-likelihood based on an assumption of normality and homogeneity while Logistic regression makes no assumption about P(X), and estimates the parameters of P(g|x) by maximizing the conditional likelihood. 
   - logistic regression would presumably be more robust if LDA’s distributional assumptions (Gaussian?) are violated. 
   - In principle, LDA should perform poorly when outliers are present, as these usually cause problems when assuming normality. 
   - In LDA, the log-membership odd between Grps are **linear functions** of the vector data x. This is due to the assumption of `Gaussian densities` and `common covariance matrices`.
   - In LogisticRegression, the log-membership odd between Grps are **linear functions** of the vector data x as well. 
   <img src="https://user-images.githubusercontent.com/31917400/52282688-e6f22300-2958-11e9-923a-5be3e22e8de9.jpg" />

## 2. Latent Dirichlet Allocation
The finite Dirichlet distribution is a distribution over distributions, namely over multinomial distributions. That means if you draw a sample from it, you get a random distribution. A loaded die can be described by a `multinomial distribution`. A machine that makes biased die with some random error can be described by a `Dirichlet distribution`. Suppose there are boxes with chocolates, with some portion of dark and sweet chocolates. You pick at random one of the boxes(perhaps some kinds of boxes can be more common than others. Then, you can pick at random one of the chocolates. So you have a distribution (a collection of boxes) of distributions (chocolates in a box). 
 - Just as the beta distribution is the conjugate prior for a binomial likelihood, the Dirichlet distribution is the conjugate prior for the multinomial likelihood. It can be thought of as a **multivariate beta distribution** for a collection of probabilities (that must sum to 1). 
   
LDA is a “generative probabilistic model” of a collection of **composites made up of parts**. 
 - The composites are `documents`.
 - The **topics** are Latent Variable. 
 - The parts are `words`  
 - `Document` is a distribution over `topics`. 
 - `Topic` is a distribution of `words`.
<img src="https://user-images.githubusercontent.com/31917400/67500637-e7635300-f67a-11e9-93f5-ff72ffa0a04b.jpg" />

The probabilistic topic model estimated by LDA consists of two tables (matrices):
 - 1st table: the probability of selecting a particular `part(word)` when sampling a particular **topic(category)**.
 - 2nd table: the probability of selecting a particular **topic(category)** when sampling a particular `document`.
<img src="https://user-images.githubusercontent.com/31917400/52525957-842abf80-2ca9-11e9-8465-b36a9e1d2d4e.jpg" />

 - > In the chart above, every topic is given the same alpha value. Each dot represents some distribution or mixture of the three topics like (1.0, 0.0, 0.0) or (0.4, 0.3, 0.3). Remember that each sample has to add up to one. At low alpha values (less than one), most of the topic distribution samples are in the corners (near the topics). For really low alpha values, it’s likely you’ll end up sampling (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), or (0.0, 0.0, 1.0). This would mean that a document would only ever have one topic if we were building a three topic probabilistic topic model from scratch.
 - > At alpha equal to one, any space on the surface of the triangle (2-simplex) is fair game (uniformly distributed). You could equally likely end up with a sample favoring only one topic, a sample that gives an even mixture of all the topics, or something in between. For alpha values greater than one, the samples start to congregate to the center. This means that as alpha gets bigger, your samples will more likely be uniform or an even mixture of all the topics.
 
 - WTF? __Simplest Generative Procedure:__
 <img src="https://user-images.githubusercontent.com/31917400/67502548-ebdd3b00-f67d-11e9-9ac6-c015878416be.jpg" />

   - Pick your unique set of WORDS.
   - Pick how many DOCUMENTS you want.
   - Pick how many WORDS you want per each DOCUMENT (sample from a Poisson distribution).
   - Pick how many `topics`(categories or label) you want.
   - Pick a number between not zero and positive infinity and call it **alpha**.
   - Pick a number between not zero and positive infinity and call it **beta**.
   - Build the `**WORDS** VS **topics** table`. 
     - For each column, draw a sample(spin the wheel) from a Dirichlet distribution using **beta** as the input. The Dirichlet distribution takes a number called **beta** for each `topic` (or category). 
     - Each sample will fill out each column in the table, sum to one, and give the probability of each part per `topic`(column).
   - Build the `**DOCUMENTS** VS **topics** table`. 
     - For each row, draw a sample from a Dirichlet distribution using **alpha** as the input. The Dirichlet distribution takes a number called alpha for each `topic` (or category).
     - Each sample will fill out each row in the table, sum to one, and give the probability of each `topic` (column) per DOCUMENT.
   - Build the actual DOCUMENTS. For each DOCUMENT:
     - Step_1) look up its **row** in the `**DOCUMENT** VS **topics** table`, 
     - Step_2) sample a `topic` based on the probabilities in the row, 
     - Step_3) go to the `**WORDS** VS **topics** table`, 
     - Step_4) look up the `topic` sampled, 
     - Step_5) sample a **WORD** based on the probabilities in the column, 
     - Step_6) repeat from step 2 until you’ve reached how many WORDS this DOCUMENT was set to have.


---------------------------------------------------------------------------------------------------------------------
## 3. AutoEncoder
 - As a **Unsupervised method**,

### Variational Inference + Neural Network = Scalable VI
10 years ago, people used to think that Bayesian methods are mostly suited for small datasets because it's computationally expensive. In the era of Big data, our Bayesian methods met deep learning, and people started to make some mixture models that has neural networks inside of a probabilistic model. 

How to scale Bayesian methods to `large datasets`? The situation has changed with the development of **stochastic Variational Inference**, trying to solve the inference problem exactly without the help of sampling. 

---------------------------------------------------------------------------------------------------------------------------------------
## > Background Knowledge: 
Let's say we have a trouble with EM in GMM...saying that we cannot calculate the `MLE value` of the soft clusters???

This is the useful story when you cannot calculate the MLE value in the EM algorithm..
<img src="https://user-images.githubusercontent.com/31917400/86541806-ce988600-bf07-11ea-8dc6-9da63e6ee9f3.jpg"/>
When MLE does not work for the original margin of log-likelihood, then we try to get a **lower bound** with the function that we can easily optimize?  Instead of maximizing the original margin of log-likelihood, we can maximize its **lower bound**!!

## Now, Let's find the Lower Bound to estimate the `MLE value`!
> [note] But it's just a lower bound.. there is no guarantee that it gives us the correct parameter estimation! 
 - Perhaps we can try...a **family of lower bounds**?? i.e. try **many different lower bounds**!
 - ## Let me introduce `q(t)` as the variational distribution of the `alpha coefficient` (mixing coefficient: probability of the hidden membership `t`= c). Here, `q(t)` are not fingers any more.  Jansen's "lower boundsss" are fingers. Any distribution can be estimated by such a bunch of "lower bounds"!!!
 - Develop a bunch of different `lower bounds`:
   - Use (0)`Hidden "t" value`, and (1)`Alpha Coefficient: q(t)`, (2) **log(**`p(x, t)/q(t)`**)**
   - **min{** `q(t)`*log[`p(x,t)/q(t)`] **}** ...KL-Divergence.. This is the Jensen's lower bound? Let's re-express our `log marginal` with KL-Divergence elements!
   - Imagine each finger(Jansen's lower bound) is a cluster ???
 <img src="https://user-images.githubusercontent.com/31917400/86539994-1fed4900-bef9-11ea-8817-ed6243b4bcbb.jpg"/>

General EM-Algorithm
<img src="https://user-images.githubusercontent.com/31917400/71264565-458b7a00-233c-11ea-88d6-e3316d5fef5b.jpg"/>
We built a lower bound on the local likelihood which depends both on the theta to maximize the local likelihood and the parameter q which is the variational distribution value, and it suggests we can optimize this lower bound in iterations by repeating the two steps until convergence. On the E-step, fix theta and maximize the lower bound with respect to q. And on the M-step, fix q and maximize the lower bound with respect of theta. So this is the general view of the expectation maximization. 
## Is it just coincidence that `Jansen's lower bound` looks like `KL-Divergence`? 
## Now we just found the first Jansen's bound as a finger. How many more fingers to go? 

------------------------------------------------------------------------------------------------------------------------
# Variational Autoencoder and Generative model: 
How can we perform efficient inference and learning in directed probabilistic models, in the presence of **continuous latent variables** with **intractable posterior distributions**, and **large datasets**? 

In contrast to the plain autoencoders, it has `sampling inside` and has `variational approximations`. 
 - for Dimensionality Reduction
 - for Information Retrieval
   
> [INTRO]: Why fitting a certain distribution into the disgusting DATA (**why do you want to model it**)?
 - If you have super complicated objects like natural images, you may want to build a probability distribution such as "GMM" based on the dataset of your natural images then try to generate **new complicated data**...
 - Application?
   - __Detect anomalies, sth suspicious__ 
     - ex> For example, you have a bank and you have a sequence of transactions, and then, if you fit your probabilistic model into this sequence of transactions, for a new transaction you can predict how probable this transaction is according to our model, our current training data-set, and if this particular transaction is not very probable, then we may say that it's kind of suspicious and we may ask humans to check it.
     - ex> For example, if you have security camera footage, you can train the model on your normal day security camera, and then, if something suspicious happens then you can detect that by seeing that some images from your cameras have a low probability of your image according to your model. 
   - __Deal with N/A__
     - ex> For example, you have some images with obscured parts, and you want to do predictions. In this case, if you have P(X) - probability distribution of your data -, it will help you greatly to deal with it. 
   - __Represent highly structured data in low dimensional embeddings__
     - ex> For example, people sometimes build these kind of latent codes for molecules and then try to discover new drugs by exploring this space of molecules in this latent space.....?? 

## Let's model the image `P(x)` ! Yes, it's about damn large sample size with high dimensionality..in the context of Unsupervised Learning.
<img src="https://user-images.githubusercontent.com/31917400/71101742-24495300-21af-11ea-9821-a14e07c54148.jpg"/>

 - [1.CNN]: Let's say that **CNN** will actually return your **logarithm of probability**. 
   - The problem with this approach is that you have to normalize your distribution. You have to make your distribution to sum up to one, with respect to sum according to all possible images in the world, and there are billions of them. So, this normalization constant is very expensive to compute, and you have to compute it to do the training or inference in the proper manner. HOW? You can use the chain rule. `Any probabilistic distribution can be decomposed into a product of some conditional distributions`, then we build these kind of conditional probability models to model our `overall joint probability`. 
 - [2.RNN]: how to represent these `conditional probabilities` is with **RNN** which basically will read your image pixel by pixel, and then outputs your prediction for the next pixel - Using proximity, Prediction for brightness for next pixel for example! And this approach makes modeling much easier because now normalization constant has to think only about 1D distribution.
   - The problem with this approach is that you have to generate your new images one pixel at a time. So, if you want to generate a new image you have to first generate X1 from the marginal distribution X1, then you will feed this into the RNN, and it will output your distribution on the next pixel and etc. So, no matter how many computers you have, one high resolution image can take like minutes which is really long...
 - ### [3. Our pick is pdf] This is very important. Let's find the density model of our data (predictive distribution)!
   - We believe `x ~ Gaussian`
   - **CNN with Infinite continuous GMM:** In short, we can try **`infinite mixture of Gaussians` which can represent any probability distribution!** Let's say if each object (image X) has a corresponding **latent variable `t`**, and the image X is caused by this **`t`**, then we can marginalize out w.r.t **`t`**, and the conditional distribution `P(X|t)` is Gaussian. We can have a mixture of infinitely many Gaussians, for each value of **"t"**(membership?). 
     - Then we mix these Gaussian with **weights**(mixing coefficients). Yes. We are trying to use Neural Network (a.k.a weighting machine) inside this model at the end... 
     - First, we should define the **prior** `P(t)` and the **likelihood** `P(x|t)`  to model `P(x)` which is the Sum( `P(x,t)`: **the un-normalized posterior** )
       - (1)`Prior` for the latent variable **t**: `P(t) = N(0, I)`.. oh yeah..the membership `t` around 0 ... **Done and Dusted**. 
       - (2)`Likelihood` for the data **x**: `P(x|t) = N( μ(t), Σ(t) )`...it can be a gaussian with parameters relying on `t`... **This is tricky**! 
         - `μ(t)` = W*`t` + b  (Of course, each component's location would be subject to the membership `t`)
         - `Σ(t)` = ![formula](https://render.githubusercontent.com/render/math?math=\Sigma_0) (Of course, each component's size would be subject to the membership `t`) 
         - REALLY???? Here we are skeptical about the above linearity of the parameterization..
           - `μ(t)` = ![formula](https://render.githubusercontent.com/render/math?math=CNN_1(t))..if you input `t`, this CNN outputs the mean? blurry? `image vector`!
           - `Σ(t)` = ![formula](https://render.githubusercontent.com/render/math?math=CNN_2(t))..if you input `t`, this CNN outputs the `Cov matirx`
           - `CNN` generate weights `w`...at the end..CNN is just giving you a bunch of weights to your likelihood..like a weighting machine. Let's say the `w` is another parameter...it's like...**mixing coefficient**?  
             - ![formula](https://render.githubusercontent.com/render/math?math=CNN_1(t)) -> `μ(t|w)`
             - ![formula](https://render.githubusercontent.com/render/math?math=CNN_2(t)) -> `Σ(t|w)`... problem is that this is too huge...
               - How about Let CNN ignores other covariance values except "diag(![formula](https://render.githubusercontent.com/render/math?math=\sigma^2(t,w)))" 
                 - `Σ(t|w)` -> "diag(![formula](https://render.githubusercontent.com/render/math?math=\sigma^2(t,w)))" 
     - Now, let's train our model! Find the partameters - `t`, `w`
       - `MLE`: Can you get some probability values for each datapoint? Let's maximize the density of our data given the parameters - `w`,`t` ? What is important is that the mixing coefficient `w` depends on `t`. If we have a latent variable, it's natural to go with Generalized EM-Algorithm, building `Jansen's bounds` on the MLE and maximize the sum of those bounds! But...you cannot imagine the analytical form of the likelihood `P(x|t)` = N( ![formula](https://render.githubusercontent.com/render/math?math=CNN_1(t,w)), ![formula](https://render.githubusercontent.com/render/math?math=CNN_2(t,w)) ). So..we can't get Sum of joints ???
         - SUM(**`log[P(x|w)]`** per each observation`x`)..so try to come up with another "SUM" caused by the latent variable `t`. 
       - `MCMC`? `VI`? ... ok, then can we obtain the un-normalized posterior:`P(x,t)`? Although knowing the prior `P(t)`, you cannot imagine the analytical form of the likelihood `P(x|t)` = N( ![formula](https://render.githubusercontent.com/render/math?math=CNN_1(t,w)), ![formula](https://render.githubusercontent.com/render/math?math=CNN_2(t,w)) ). So..we can't get Sum of joints???
       - Anyway, we decide our predictive null model - mean(x) - is explained by Neural Network.... 
       - Then how to train? How to find the parameter - `t`,`w` - in the first place?
       
## You know what? we are start off with Decoder ?
Only if we have `hidden variables`...
   <img src="https://user-images.githubusercontent.com/31917400/72342676-f3344380-36c4-11ea-90a2-ea05caf2e11a.jpg"/>
 
   - ## Overview: get some variational distribution `q(t)` or `LB(t)` ?
   <img src="https://user-images.githubusercontent.com/31917400/86674224-5bb70a00-bff0-11ea-91d6-c6907d62ae6a.jpeg"/>
   
   - *In VI, the **KL-Divergence** (where each `q(z)` is a `finger`) should be **`minimized`**. In VI, the **MLE estimator** is the joint of all `q(z)`,
   - *In VAE, the `Jansen's lower bound` as a **KL-Divergence** needs to be **`maximized`**..(where `q(z)` is a mixing coefficient for GMM form with `log(p/q)` as a Gaussian cluster) and each Jansen's lower bound is a `finger`. In VAE, the **MLE estimator** is the sum of a bunch of `Jansen's lower bound`ssss. 
   
   - __[Story]:__ `**Encoding**: Discover the latent parameter from our dataset` -> `**Decoding**: Generate new data based on the latent memberships`
     - Ok, let's do some reverse enigineering. **Back to the question. How to train?** How to find the parameter in the first place?
     - ## `t` and `w` is the key!
       - Let's try **`Variational Inference`**. Assuming each **q(![formula](https://render.githubusercontent.com/render/math?math=t_i))** as the Exponential family function = N(![formula](https://render.githubusercontent.com/render/math?math=m_i), ![formula](https://render.githubusercontent.com/render/math?math=s_i^2)), so each `q(t)` is different Gaussian...and the value is probability as a mixing coefficient. Then we can **maximize the Jansen's Lower Bound** w.r.t `q`, `m`, `s^2`. But it's so complicated..Is there other way? -> VAE... 
       - Both `t`,`w` are parameters for Decoding...our final predictive model. 
       - The solution is "Encoding" since it returns the distribution of `t`. Remember? `w`(mixing coefficient) relies on `t`(membership). 
       - Hey, so we first want to obtain the latent variable space! We are conjuring the **Encoder** that outputs the latent parameter `t` space since `w` results from `t`. Let's find the posterior `P(t|x)` as an Encoder..then we would someday get `P(x|t)`as a Decoder. 
         - __[Find `t`]__ **Bring up the "factorized" variational distribution `q(t)`** and let NN return (![formula](https://render.githubusercontent.com/render/math?math=m_i), ![formula](https://render.githubusercontent.com/render/math?math=s_i^2)) that explains the distribution of `t` **which is a Gaussian** and we call it `q(t)` function ... we can say that the latent variable `t` follows Gaussian.    
           -: Let's make `q(t)`= N(m, s^2) flexible. If assume all **q(![formula](https://render.githubusercontent.com/render/math?math=t_i))** = N( ![formula](https://render.githubusercontent.com/render/math?math=CNN_1(t))=`m(x_i, φ)`, ![formula](https://render.githubusercontent.com/render/math?math=CNN_2(t))=`s^2(x_i, φ)` ), then the training get much easier. Since we already have the original input data `x`, we can simply ask CNN to produce weight `φ`.
           <img src="https://user-images.githubusercontent.com/31917400/86669433-7dfa5900-bfeb-11ea-9160-c33cde0b9c08.jpg"/>
           
           -: Once we pass our initial data `x` through our [first neural network] as an encoder with parameters`φ`, it returns `m`,`s^2` which are parameters of `q(t)`. How `t` are distributed? Normally...
             - ## We found `q(t)` = `P(t|x)` = `N(m, s^2)` which is an unique mixing coefficient function...Interestingly, we forget about the mixing coefficient and simply do MonteCarlo Sampling from this distribution`q(t)` to get random data pt `t`.  
         - __[Find `w`]__ Now, we know `t` so we can get `w`! Let's pass this sampled vector `T` into the `second neural network` to get parameters`w`. 
           -: It outputs us the distribution that are as close to the input data as possible.
           <img src="https://user-images.githubusercontent.com/31917400/86837661-1d285a80-c097-11ea-936f-8dbafdce6945.jpg"/>
           
     - ## Next, how to define the CNN's weighting mechanism for `Φ` and `w` ? : Maximize Jensen's Lower bound in the gradient for CNN! 
       - **`Latent variable distribution: q(t)` is useful!** Anomaly Detection for a new image which the network never saw, of some suspicious behavior or something else, our conditional neural network of the encoder can output your **latent variable distribution** as far away from the Gaussian. By looking at the distance between the variational distribution `q(t)` and the standard Gaussian, you can understand how anomalistic a certain point is ... wait. `P(t)` is Standard Normal?   
       <img src="https://user-images.githubusercontent.com/31917400/72226852-bca7dd00-358d-11ea-98d6-20965d0dce46.jpg"/>
   
       - __Gradient of Encoder:__ Make an Expected Value ?
         - we're passing our image through our Encoder, and compute the **usual gradient** of this first neural network with respect to its parameters `Φ` to get the parameters(Φ) of the variation distribution `q(t|Φ)`. We use **"log derivitive trick"** to approximate the gradient (make the form of expected value?) but it has some problem: `the variance of this stochastic approximation will be so high that you will have to use lots and lots of gradients to approximate this thing accurately`. How can we estimate this gradient with a much **smaller variance estimate**?  
       - __Gradient of Decoder:__ Make an Expected Value ?
         - we sample `t` from the variation distribution `q(t|Φ)` and put this `point` as input to the Decoder with parameters `w`. And then we just compute the **usual gradient** of this second neural network with respect to its parameters `w`.  
       <img src="https://user-images.githubusercontent.com/31917400/72433990-7a9bb880-3792-11ea-8cfd-f3e6778fa8ad.jpg"/>
   
       - __Issues of gradient of Encoder:__ 허벌창 그라디언트여? How can we better estimate this varying gradient with a much **smaller variance estimate**?  
         - 왜 허벌창? our input data (x) is 이미지니깐...
         - when sampling `t`, **"reparameterization trick"** of our latent variable makes the a Jensen's lower bound estimator easy to be optimized using standard stochastic gradient.
         - so..you just sample from a identity matrix...All works will be done by `m` and `s^2`..
       <img src="https://user-images.githubusercontent.com/31917400/73176973-afe6c580-4105-11ea-8822-49b2d202c156.jpg"/>


### Learning with priors
<img src="https://user-images.githubusercontent.com/31917400/69436481-5b0b8500-0d39-11ea-8e3d-1d565674042e.jpg"/>

### EX> Variational Dropout and Scalable BNN
Compress NN, then fight severe overfitting on some complicated datasets. 
 
We first pick a fake? posterior `q(z|v)` as a **family of distributions** over the `latent variables` with **its own variational parameters**`v`. KL-divergence method helps us to minimize the distance between `P(z)` and `q(z)`, and in its optimization process, we can use `mini-batching` training strategy(since its likelihood can be split into many pieces of log sum), which means we don't need to compute the whole training of the likelihood. ELBO supports mini-batching.    
 - We can use MonteCarlo estimates for computing stochastic gradient, which is especially useful when the reparameterization trick for `q(z|v)` is applicable. 
 
????????????????????????????????????????????????????????????????? 


































































