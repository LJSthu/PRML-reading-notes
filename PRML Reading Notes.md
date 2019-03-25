> 刘家硕

# Prologue

Machine learning is a promising area of huge contents, closely related to many fields such as calculus, linear algebra, probability theory, statistics and so on. As a beginner, I can not memorize all things. So I wrote down some thoughts and ideas through my reading process, which can help me gain a profound understanding of machine learning, especially from the viewpoint of probability. 

Since this note was not for publication but to help me review the book and important theorems, sometimes I just copied some formulas, theorems, sentences from the Pattern Recognition and Machine Learning to save time. All copyright is owned by corresponding authors. 


This note contains:
> [1] important theorems and corresponding derivations
> [2] my own thoughts
> [3] corrections of some mistakes made in the book
> [4] supplements of the derivation for some formulas

# 1. Introduction
This chapter served as an introduction to the pattern recognition and machine learning. Taking the polynomial curve fitting as an example, it discussed something fundamental in pattern recognition, such as probability theory, frequency and Bayesian probabilities, overfitting problems, dimensionality curse, decision theory and information theory. I summarized what impressed me most in the following.

## 1.1 Polynomial Curve Fitting
Polynomial curve fitting problem leads to MLE, MAP and Bayesian methods, which are quite important in pattern recognition and machine learning.

As for the maximum likelihood estimation, given the train data $X$, we maximize the likelihood function to find the best parameter $\theta$: 
$$
\begin{equation}
\hat{\theta} = arg max_{\theta} \sum_{i=1}^N p(x_i|\theta)
\end{equation}
$$
This corresponds to choosing the value of $\theta$ for which the probability of the observed data set is maximized. As for the polynomial curve fitting, we can also define a loss function and minimize the loss function to gain the $\theta$:
$$
\begin{equation}
\hat{\theta} = arg min_{\theta} \sum_{i=1}^N (y(x_i, \theta) - t_i)^2
\end{equation}
$$
Interestingly, we can find that maximization of the likelihood function under a conditional Gaussian noise distribution for a linear model is equivalent to minimizing a sum-of-squares error function, which shows the relationship between MLE and the square loss.  

The **Maximum A Posterior** method focuses on the posterior distribution. Using Bayes’ theorem, the posterior distribution for $\theta$ is proportional to the product of the prior distribution and the likelihood function:
$$
\begin{equation}
p(\theta|X, t, \alpha) \propto p(t|X, \theta)p(\theta| \alpha)
\end{equation}
$$ where $\alpha$ controls the distribution of parameters, called hyperparameters. 


We can now determine $\theta$ by finding the most probable value of $\theta$ given the data, in other words by maximizing the posterior distribution. When minimizing the square loss function to solve this problem, we often face the overfitting problem, which can be avoided by regularization. But I am confused why adding a regularization component works. To some extent, we can understand it in Bayesian way. After calculating, we can see that maximizing the posterior distribution is equivalent to minimizing the regularized sum-of-squares error function, which impressed me a lot. And this is the first time for me to feel the charm of Bayesian methods.

However, MLE and MAP are still making a point estimate of $\theta$ and are not a fully Bayesian approach. In a fully Bayesian approach, we should consistently apply the sum and product rules of probability, which requires integrating over all values of $\theta$. 
$$
\begin{equation}
p(t|x, X, t) = \int p(t|x, \theta)p(\theta|X, t)d\theta
\end{equation}
$$
The formula above is to predict the label given a new-coming data in a Bayesian way. Such marginalizations lie at the heart of Bayesian methods for pattern recognition and will show during the whole book. 


## 1.2 Model Selection
Cross-validation can be used to avoid the over-fitting problem and select the model. A special case of cross-validation is called **leave-one-out** , in which the size of the validation set is one, and it is used when the dataset is quite small. Cross-validation is time-consuming especially when the dataset is large. This method is a frequentist approach and the following is a Bayesian approach. 


There are many information criterions raised for this. Akaike information criterion chooses the model which maximize the:
$$
lnp(D|W_{ML}) - M
$$ where $M$ is the number of the adjustable parameters. It has something to do with Bayesian information criterion(BIC). 


Bayesian approaches can to some extent elude over-fitting problems. Marginalize the parameters out to leave the hyperparameters only, we can compare the magnitudes of evidence functions under different hyperparameters to select the best model.

## 1.3 Decision Theory
This part introduced some basic knowledge about decision theory. It is quite important to differentiate the discriminative models and the generative models, which are two kinds of models in classification problems. As for the generative model, we would like to model the joint distribution $ p(x, C_k )$ directly and then normalize to obtain the posterior probabilities. However, a discriminative model only focuses on determining the posterior class probabilities $p(C_k|x)$.

### 1.3.1 Minimizing the misclassification rate
The goal is simply to make as few misclassifications as possible. For a k-classification problem, it is to maximize the following:
$$
p(correct) = \sum_{k=1}^Kp(x \in R_k, C_k) = \sum_{k=1}^K \int_{R_k}p(x,C_k) dx
$$
The maximum is obtained when $x$ is assigned to the class whose the posterior probability $p(C_k|x)$ is largest. 

### 1.3.2 Minimizing the expected loss
The average loss is given by:
$$
E[L] = \sum_k \sum_j \int_{R_j} L_{kj}p(x, C_k)dx
$$ where $L_{kj}$ is the loss when we assign an input $x$ to class $C_j$ when its real class is $C_k$.

### 1.3.3 The reject option
In a classification problem, when the largest posterior probability $p(C_k|x)$ is  significantly less than unity, it will be appropriate to avoid making decisions. The model should know when to reject the decision instead of giving a n answer which is probably wrong. 

### 1.3.4 Loss functions for regression
The average loss function is given by:
$$
E[L] = \int \int L(t,y(x))p(x,t)dx dt
$$
We can employ calculus of variations to find the $y(x)$ by minimizing the above equation. 
$$
y(x) = E_t[t|x]
$$

## 1.4 Information Theory
The amount of information can be viewed as the ’degree of surprise’ on learning the value of $x$. Low probability events $x$ correspond to high information content. The entropy of the random variable $x$ is calculated as
$$
H[x] = -\sum_xp(x)log_2p(x)
$$

The nonuniform distribution has a smaller entropy than the uniform one, which can be viewed by the average code length. The <cite>noiseless coding theorem (Shannon, 1948)</cite> states that the entropy is a lower bound on the number of bits needed to transmit the state of a random variable. The differential entropy is defined as:
$$
H[x] = -\int p(x)lnp(x)dx
$$
Note that the discrete and continuous forms of the entropy differ by a quantity $ln\delta$, which shows that to specify a continuous variable requires a large number of bits. We can maximize the differential entropy with the three constraints(the first and second moments of $p(x)$ as well as preserving the normalization constraint). We can find that the distribution that maximizes the differential entropy is the Gaussian. And the differential entropy unlike the discrete entropy, can be negative. The conditional entropy of $y$ given $x$ is defined as:
$$
H[y|x] = -\int y(x,y)ln(p(y|x))dy dx
$$
And conditional entropy satisfies the relation below:
$$
H[x,y] = H[x] + H[y|x]
$$
Thus the information needed to describe $x$ and $y$ is given by the sum of the information needed to describe $x$ alone plus the additional information required to specify $y$ given $x$.
$$
KL(p||q) = -\int p(x)ln\frac{q(x)}{p(x)}dx
$$
This is known as the relative entropy or <cite>Kullback-Leibler divergence</cite>, or KL divergence (Kullback and Leibler, 1951), between the distributions $p(x)$ and $q(x)$. Note that it is not a symmetrical quantity. The mutual information between the variables $x$ and $y$ is calculated as:
$$
I[x,y] = -\int \int p(x,y) ln(\frac{p(x)p(y)}{p(x,y)})dx dy
$$

## 1.5 Frequentist methods and Bayesian methods
There are two different interpretations related to probability, frequentist and Bayesian. In the eyes of Bayesian statisticians, probability is an approach to handle uncertainty instead of a measure related to frequency. In my opinion, the author focused on Bayesian view.


From the polynomial curve fitting problem, we can see that the Bayesian methods can avoid the over-fitting problem of maximum likelihood since we have averaged all possibilities of model complexities. When using Bayesian methods, we have to choose a suitable prior. But the prior is often selected for mathematical convenience rather than reflecting the properties of the training data, which is a disadvantage of Bayesian methods. The prior is the knowledge we give to the model. If we can find out an suitable way to add knowledge to the model, the learning procedure may be much more efficient.

# 2. Probability Distributions
This chapter discuss some particular examples of probability distributions and their properties, which form building blocks for more complex models and will be used extensively throughout the book. Cause I was familiar with this, I just summarize several probability distributions and their properties. Here I referred to <cite>Yang Song's notes</cite> to organize the probability distributions in the conjugate priors part.

## 2.1 Conjugate Priors
Conjugate priors lead to posterior distributions having the same functional form as the prior, and that therefore lead to a greatly simplified Bayesian analysis.

### 2.1.1 Bernoulli distribution
$$
Bern(x|\mu) = \mu_x(1-\mu)^{1-x}
$$ where $x = \{0,1 \}$ and $p(x = 1|\mu) = \mu$
Given $\mu$ and data $D$, the likelihood funciton is as following:
$$
p(D|\mu) = \prod_{n = 1}^Np(x_n|\mu) = \prod_{n=1}^N \mu^{x_n}(1-\mu)^{1-x_n}
$$

### 2.1.2 Binomial distribution
$$
Bin(m|N, \mu) = C_N^m\mu^m(1-\mu)^{N-m}
$$
The **conjugate distribution** of binomial distribution is **beta** distribution. 

### 2.1.3 Beta distribution
$$
Beta(\mu|a,b) = \frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)}\mu^{a-1}(1-\mu)^{b-1}
$$ where $\Gamma(x)$ is called Gamma function. $a,b$ are often called hyperparameters, since they control the distribution of $\mu$.

Using beta distribution as the prior, the posterior distribution of $\mu$ is as following:
$$
p(\mu|m,l,a,b) \propto \mu^{m+a-1}(1-\mu)^{l+b-1}
$$ where $m+l=N$. 

### 2.1.4 Multinomial distribution
Binary variables can be used to describe quantities that can take one of two possible values. When dealing with discrete variables that can take on one of K possible mutually exclusive states, a particularly convenient representation is the 1-of-K scheme. The variable is represented by a K-dimensional vector $x$ in which one of the elements $x_k$ equals 1, and all remaining elements equal 0.  If we denote the probability of $x_k = 1$ by the parameter $\mu_k$, then the distribution of $x$ is given by
$$
p(x|\mu) = \prod_{k=1}^K\mu_k^{x_k}
$$ where $\sum_k\mu_k = 1$
Consider a data set $D$ of $N$ independent observations, the likelihood function is 
$$
p(D|\mu) = \prod_{n=1}^N \prod_{k=1}^K\mu_k^{x_{nk}} = \prod_{n=1}^N \prod_{k=1}^K\mu_k^{mk}
$$ where $m_k = \Sigma_nx_{nk}$ which is called the sufficient statistics. 
$$
Mult(m_1, m_2, ..., m_K|\mu, N) = \frac{N!}{m_1!m_2!...m_K!} \Pi_{k=1}^K \mu_k^{m_k}
$$
The conjugate prior is Dirichlet distribution. 

### 2.1.5 Dirichlet distribution
$$
Dir(\mu|\alpha) = \frac{\Gamma(\alpha_0)}{\Gamma(\alpha_0)...\Gamma(\alpha_K)}\prod_{k=1}^K \mu_k^{\alpha_k-1}
$$ where $\alpha_0 = \Sigma_{k=1}^K\alpha_k $
Using Dirichlet distribution as the prior, the posterior distribution is given as:
$$
p(\mu|D, \alpha) \propto \prod_{k=1}^K\mu_k^{\alpha_k+m_k-1}
$$


### 2.1.6 Gaussian distribution and its conjugate priors
The Gaussian distribution takes the form
$$
\begin{equation}
	\cal{N}(\bf{x}|\boldsymbol{\mu,\Sigma})=\frac{1}{(2\pi)^{D/2}}\frac{1}{|\boldsymbol{\Sigma}|^{1/2}}\exp\left\lbrace-\frac{1}{2}(\bf{x}-\boldsymbol{\mu})^{\intercal}\boldsymbol{\Sigma}^{-1}(\bf{x}-\boldsymbol{\mu}) \right\rbrace 
\end{equation}
$$
where $\boldsymbol{\Sigma}$ is a symmetric and positive definite covariance matrix. 

The likelihood function is:
\begin{equation}
	p(\bf{X}|\mu)=\prod_{n=1}^{N}p(x_n|\mu)=\frac{1}{(2\pi \sigma^2)^{N/2}}\exp\left\lbrace-\frac{1}{2\sigma^2}\sum_{n=1}^{N}(x_n-\mu)^2\right\rbrace
\end{equation}

Fixing variance and varying mean, the conjugate prior is a Gaussian distribution. Fixing mean and varying variance, the conjugate prior is a Gamma distribution, which is defined as
$$
\begin{equation}
	{Gam}(\lambda|a,b)=\frac{1}{{\Gamma}(a)}b^a \lambda^{a-1}\exp(-b\lambda)
\end{equation}
$$
where $\lambda$ denotes the **precision**, e.g. $\lambda = 1/\sigma^2$.

If both the mean and variance are unknown, the conjugate prior is Gaussian-Gamma distribution
$$
\begin{equation}
	p(\mu,\lambda)=\cal{N}(\mu|\mu_0,(\beta\lambda)^{-1}){Gam}(\lambda|a,b)
\end{equation}
$$

## 2.2 The Gaussian Distribution
The Gaussian, also known as the normal distribution, is a widely used model for the distribution of continuous variables.

### 2.2.1 Conditional and marginal Gaussian distribution
Given a joint Gaussian distribution $\cal{N}(\bf{x}|\boldsymbol{\mu},\boldsymbol\Sigma)$ with $\boldsymbol{\Lambda\equiv \Sigma^{-1}}$ and

\begin{equation}
 {x}=\left( \begin{array}{c}
 {x}_a \\ {x}_b
\end{array} \right) ,\qquad \boldsymbol{\mu}=\left(\begin{array}{c}
\boldsymbol{\mu}_a\\
\boldsymbol{\mu}_b
\end{array}\right)
\end{equation} 
\begin{equation}
\boldsymbol{\Sigma}=\boldsymbol{\begin{pmatrix}
\Sigma_{aa} & \Sigma_{ab} \\
\Sigma_{ba} & \Sigma_{bb}
\end{pmatrix}},\qquad 
\boldsymbol{\Lambda=\begin{pmatrix}
\Lambda_{aa} & \Lambda_{ab}\\
\Lambda_{ba} & \Lambda_{bb}
\end{pmatrix}}
\end{equation}
The conditional Gaussian distribution is given as:
$$
\begin{align}
p({x}_a|{x_b})&=\cal{N}({x}_a|\boldsymbol{\mu}_{a|b},\boldsymbol\Lambda_{aa}^{-1})\\
\boldsymbol{\mu}_{a|b}&=\boldsymbol \mu_a - \boldsymbol{\Lambda}_{aa}^{-1}\boldsymbol{\Lambda}_{ab}({x}_b-\boldsymbol{\mu}_b)
\end{align}
$$
And the marginal distribution is:
$$
\begin{equation}
	p({x}_a)=\cal{N}({x}_a|\boldsymbol{\mu}_a,\boldsymbol{\Sigma}_{aa})
\end{equation}
$$


### 2.2.2 Student’s t-distribution
The conjugate prior for the precision of a Gaussian is given by a gamma distribution. If we have a univariate Gaussian $\cal{N}(x|\mu,\tau^{-1})$ together with a Gamma prior ${Gam}(\tau|a,b)$ and we integrate out the precision, we obtain the marginal distribution of $x$ in the form
\begin{equation}
	p(x|\mu,a,b)=\int_{0}^{\infty} \cal{N}(x|\mu,\tau^{-1}){Gam}(\tau|a,b)  d \tau
\end{equation}
Defining $\nu = 2a$ and $\lambda=a/b$, the distribution $p(x|\mu,a,b)$ is as following:
\begin{equation}
	{Student}(x|\mu,\lambda,\nu)=\frac{{\Gamma}(\nu/2+1/2)}{{\Gamma}(\nu/2)}\left(\frac{\lambda}{\pi \nu}\right)^{1/2}\left[1+\frac{\lambda(x-\mu)^2}{\nu}\right]^{-\nu/2-1/2}
\end{equation} where $\lambda$ is called the precision of the distribution. 
Student's t-distribution can be viewed as the sum of infinite Gaussian distribution with the same mean but different precisions. Compared to Gaussian, Student's t-distribution is more robust to outliers, which can be seen in the following graph.
![屏幕快照 2019-03-04 下午2.30.02.png](resources/119663747710CE6A06AF806676951074.png =746x459)


### 2.2.3 Mixtures of Gaussians
Mixture distributions are formed by linear combinations of more basic distributions such as Gaussians. Though Gaussian distribution is mostly used, it it is insufficient to fit some complex distributions. However, by using a sufficient number of Gaussians, and by adjusting their means and covariances as well as the coefficients in the linear combination, almost any continuous density can be approximated to arbitrary accuracy.
![屏幕快照 2019-03-04 下午2.37.06.png](resources/4E16722BF3F2848A79C2E8BFB60CCF6F.png =570x177)
The model is defined as:
\begin{equation}
p(x) = \sum_{k=1}^K\pi_k \cal{N}(x|\mu_k, \Sigma_k)
\end{equation} and each Gaussian $\cal{N}(x|\mu_k, \Sigma_k)$ in it is called a component with its own mean and covariance. 
The likelihood is :
\begin{equation}
	ln p(\bf{X}|\pi, \mu, \Sigma)=\sum_{n=1}^{N}ln\left\lbrace \sum_{k=1}^K \pi_k\cal{N}(x_n|\mu_k, \Sigma_k)\right\rbrace
\end{equation}
The situation is now much more complex than with a single Gaussian, due to the presence of the summation over $k$ inside the logarithm. 

### 2.2.4 Linear Gaussian model
A linear Gaussian model comprises of a Gaussian marginal distribution $p({x})$ and a Gaussian conditional distribution $p(y|x)$ in which $p({y}|{x})$ has a mean that is a linear function of ${x}$, and a covariance which is independent of ${x}$.

We take the marginal and conditional distributions to be
\begin{align}
	p({x}) &= \cal{N}({x}|\boldsymbol{\mu,\Lambda^{-1}})\\
	p({y}|{x}) &= \cal{N}({y}|{Ax}+{b},\bf{L}^{-1})
\end{align}
the marginal distribution of ${y}$ and the conditional distribution of ${x}$ given ${y}$ are given by
\begin{align}
	p({y})&=\cal{N}({y}|\bf{A}\boldsymbol{\mu}+{b},\bf{L}^{-1}+\bf{A}\boldsymbol{\Lambda}^{-1}\bf{A}^{\intercal})\\
	p({x}|{y})&=\cal{N}({x}|\boldsymbol{\Sigma}\{\bf{A}^{\intercal} {L(y-b)}+\boldsymbol{\Lambda\mu}\},\boldsymbol{\Sigma})
\end{align}
where 
\begin{equation}
	\boldsymbol\Sigma = (\boldsymbol{\Lambda}+\bf{A}^{\intercal}\bf{LA})^{-1}
\end{equation}


## 2.3 The Exponential Family
Given parameter $\eta$, the exponential family of distributions over $x$ is defined as the set of distributions of the following form:
\begin{equation}
	p(x|\boldsymbol{\eta})=h({x})g(\boldsymbol{\eta})\exp\{\boldsymbol{\eta}^{\intercal}{u(x)}\}
\end{equation} where $\eta$ has no restrictions. $\boldsymbol{\eta}$ is called the natural parameters of the distribution. 
The function $g(\boldsymbol{\eta})$ can be interpreted as the coefficient that ensures that the distribution is normalized and therefore satisfies:
\begin{equation}
g(\boldsymbol{\eta})\int h(x)\exp\{\boldsymbol{\eta}^{\intercal}{u(x)}\} dx = 1
\end{equation}
Note that the Bernoulli distribution, Binomial distribution, Gaussian distribution are all members of exponential family (except Gaussian Mixture Distribution).

### 2.3.1 Maximum likelihood and sufficient statistics
Taking the gradient of both sides of the above equation with respect to $\eta$, we have:
$$
\begin{equation}
	-\nabla \ln g(\boldsymbol \eta) = \mathbb{E}[\bf{u(x)}]
\end{equation}
$$
From the maximization of the likelihood function, 
\begin{equation}
p(\bf{X}|\eta) = (\prod_{n=1}^Nh(x_n))g(\eta)^N\exp \left \lbrace \eta^{\intercal}\sum_{n=1}^Nu(x_n) \right\rbrace
\end{equation}
we obtain
\begin{equation}
	-\nabla \ln g(\boldsymbol{\eta}_{ML}) = \frac{1}{N} \sum_{n=1}^{N} \bf{u(x}_n)
\end{equation} where $\sum_nu(x_n)$ is called the sufficient statistic. 
As for the Bernoulli distribution, $u(x_n)$ is equal to $x$, and for Gaussian distribution, it is equal to $(x,x^2)^{\intercal}$. We do not need to store the whole dataset, but only to store the sum of $x$ and $x_2$. That is why it is called the **sufficient statistic**. 

### 2.3.2 Conjugate priors
For any member of the exponential family, there exists a conjugate prior that can be written in the form
\begin{equation}
	p(\boldsymbol{\eta}|\boldsymbol{\chi},\nu)=f(\boldsymbol{\chi},\nu)g(\boldsymbol{\eta})^{\nu}\exp\{\nu \boldsymbol{\eta^{\intercal}\chi}\}
\end{equation}

### 2.3.3 Non-informative Priors
Here we use two simple examples from <cite>Berger, 1985</cite>
For a translation invariant density taking the form
\begin{equation}
	p(x|\mu)=f(x-\mu)
\end{equation}
the non-informative prior $p(\mu)$ is constant.

For a scale invariant density taking the form
\begin{equation}
	p(x|\sigma)=\frac{1}{\sigma} f(\frac{x}{\sigma})
\end{equation}
the non-informative prior satisfies $p(\mu) \propto 1/\sigma$.

Note that the prior may be improper, namely being unable to be normalized. However, if the posterior distribution can be normalized, improper priors can still be applied.


## 2.4 Nonparametric Methods
The models we have discussed throughout this chapter are called the parametric approach to density modelling. An important limitation of this approach is that the chosen density might be a poor model of the distribution that generates the data, which can result in poor predictive performance. Here goes to the nonparametric methods. 

### 2.4.1 Kernel density estimators
Let us suppose that observations are being drawn from some unknown probability density $p(x)$ in some D-dimensional space, which we shall take to be Euclidean, and we wish to estimate the value of $p(x)$. The density estimate takes the form \begin{equation}
    p(x) = \frac{K}{NV}
\end{equation}
Divide data space into hypercubes of volume $h^{D}$. Define a kernel function $k$ to measure similarities between data vectors. Then the estimator takes the form
\begin{equation}
	p(\bf{x}) = \frac{1}{N} \sum_{n=1}^{N} \frac{1}{h^{D}}k(\frac{\bf{x-x}_n}{h})
\end{equation}
where $h$ plays the role of a hyperparameter.

Gaussian kernel is among one of the most frequently used kernels, which has the form:
\begin{equation}
p(x) = \frac{1}{N}\sum_{n=1}^N\frac{1}{(2\pi h^2)^{\frac{D}{2}}}\exp\left\lbrace -\frac{|x-x_n|^2}{2h^2}\right\rbrace
\end{equation}


### 2.4.2 Nearest-neighbor methods
To estimate the density $p(\bf{x})$, we allow the radius of the hypersphere centered at $\bf{x}$ to grow until it contains precisely $K$ data points.

The K-nearest-neighbor technique can be extended to classification problems making use of Bayes' theorem.
\begin{equation}
	p(\cal{C}_k|{x})=\frac{p({x}|C_k)p(\cal{C}_k)}{p({x})}=\frac{K_k}{K}
\end{equation} since $p(x) = \frac{K}{NV}$ and $p(x|C_k) = \frac{K_k}{N_kV}$

# 3. Mathematics 

## 3.1 Matrix

## 3.2 Optimizing

## 3.3 Appromixmate Methods

## 3.4 Sampling Methods

# 4. Linear Models
First we focus on the linear models, which are often expressed as:
\begin{equation}
y(\bf{x}, \bf{w}) = \bf{w}^{\intercal}x
\end{equation}
The key property of this model is that it is a linear funcition of the parameters $w_0, w_1, ..., w_D$. 

## 4.1 Linear Models for Regression
Given $bf{f(x)}$, the observed target variable $t$ obeys the following distribution:
\begin{equation}
	p(t|\bf{x,w},\beta) = \cal{N}(t|y(\bf{x,w}),\beta^{-1}). \label{LinearR}
\end{equation}
Note that in linear regression problems, we are not trying to model the distribution of the input variables. Thus $\bf{x}$ will always appear in the set of conditioning variables. More specifically, the model we discussed here is a discriminative model rather than a generative model. As a result, we just model the conditional distribution where $bf{x}$ is observed and serves as the condition.  

### 4.1.1 Maximum likelihood and least squares
Assume the target variable $t$ is given by a deterministic function $y(x, \bf{w})$ with additive Gaussian noise:
\begin{equation}
t = y(x, \bf{w}) + \epsilon
\end{equation} where $\epsilon \in \boldsymbol{N} (\epsilon|0, \beta^{-1})$
Therefore, we have the likelihood function:
\begin{equation}
p({t}|\bf{X,w},\beta) = \prod_{n=1}^N \cal{N}(t_n|\bf{w}^{\intercal} {\phi}(\bf{x}_n),\beta^{-1})
\end{equation}
In the <cite>PRML</cite>, the $x$ was dropped from expressions such as $p(t|X, \bf{w}, \beta)$ in order to keep the notation uncluttered, while I found it sometimes confused and kept it.
Taking the logarithm of the likelihood function, we have:
\begin{align}
	\ln p(\bf{t}|X, \bf{w},\beta) &= \sum_{n=1}^N \ln \cal{N}(t_n|\bf{w}^{\intercal} {\phi}(\bf{x}_n),\beta^{-1}) \notag \\
	&= \frac{N}{2} \ln \beta - \frac{N}{2} \ln(2\pi) - \beta E_D(\bf{w})
\end{align} where the sum-of-squares error function is defined as:
\begin{equation}
	E_D(\bf{w}) = \frac{1}{2} \sum_{n=1}^N \{ t_n - \bf{w}^{\intercal} {\phi}(\bf{x}_n) \}^2.
\end{equation}

The gradient of the log likelihood function takes the form
\begin{equation}
	\nabla \ln p(\bf{t}|X, \bf{w},\beta) = \beta \sum_{n=1}^N \{ t_n-\bf{w}^{\intercal} {\phi}(\bf{x}_n) \} {\phi}(\bf{x}_n)^{\intercal}.
\end{equation}
Setting this gradient to zero and solving for $\bf{w}$ we obtain:
\begin{equation}
	\bf{w}_{ML} = ({\Phi}^{\intercal} {\Phi})^{-1} {\Phi}^{\intercal} \bf{t}
\end{equation} which is the normal equation of the least squares problem. 

Here ${\Phi}$ is called the design matrix, whose elements are given by $\Phi_{nj} = \phi_j (\bf{x}_n)$. The quantity
\begin{equation}
	{\Phi}^{\dagger} \equiv ({\Phi}^{\intercal} {\Phi})^{-1} {\Phi}^{\intercal}
\end{equation}
is known as the Moore-Penrose pseudo-inverse of the matrix ${\Phi}$.
Note that if $\Phi$ is square and inverse, we have $\Phi^{\dagger} = \Phi^{-1}$
As for the bias $w_0$, it compensates for the difference between the averages (over the training set) of the target values and the weighted sum of the averages of the basis function values.
![屏幕快照 2019-03-04 下午6.49.44.png](resources/2D582C927303796B053BA1ED88E40DD8.png =577x153)
We can also maximize the log likelihood function with respect to the noise precision parameter $\beta$, giving:
\begin{equation}
	\frac{1}{\beta_{ML}} = \frac{1}{N} \sum_{n=1}^N \{ t_n - \bf{w}_{ML}^{\intercal} {\phi}(\bf{x}_n) \}^2
\end{equation} which shows that the inverse of the noise precision is given by the residual variance of the target values around the regression function.


If we solve the MLE problem as above, we will have to process the whole dataset at one time, which require huge computation. When the dataset is huge enough, we can use sequential learning method to solve this problem. For the sum-of-dquares error function, we have:
\begin{equation}
	\bf{w}^{(\tau+1)} = \bf{w}^{(\tau)} + \eta (t_n - \bf{w}^{(\tau)T} \boldsymbol{\phi}_n) \boldsymbol{\phi}_n
\end{equation}
where $\boldsymbol{\phi}_n = \boldsymbol{\phi}(\bf{x}_n)$. This is known as least-mean-squares or the LMS algorithm.

### 4.1.2 Regularized least squares
If we use regularization to avoid overfitting problem, the total loss function takes the form:
\begin{equation}
	E_D(\bf{w}) + \lambda E_W(\bf{w})
\end{equation} where $\lambda$ is the regularization coefficient that controls the relative importance of the data-dependent error $E_D(\bf{w})$ and the regularization term $E_W(\bf{w})$.
A simple choice of regularization can be:
\begin{equation}
	E_W(\bf{w}) = \frac{1}{2} \bf{w}^{\intercal}w
\end{equation}
A more general form of the regularization is:
\begin{equation}
	E_W(\bf{w}) = \frac{1}{2} \sum_{j=1}^M |w_j|^q.
\end{equation}
When $q=2$, this becomes the weight decay in machine learning, because in sequential learning algorithms, it encourages weight values to decay towards zero, unless supported by the data. Setting the gradient with respect to $\bf{w}$ to zero, we have:
\begin{equation}
w = (\lambda I + \Phi^{\intercal}\Phi)^{-1}\Phi^{\intercal}t
\end{equation}

The case of $q = 1$ is known as the lasso in the statistics literature. It has the property that if $\lambda$ is sufficiently large, some of the coefficients $w_j$ are driven to zero, leading to a sparse model in which the corresponding basis functions play no role.


### 4.1.3 The Bias-Variance Decomposition
This part we consider a frequentist viewpoint of the model complexity issue, known as the bias-variance trade-off. Suppose we choose the squared loss function, for which the optimal prediction is given by the conditional expectation, which we denote by $h(x)$ and which is given by:
\begin{equation}
h(x) = \int tp(t|x)dt
\end{equation}
The expected squared loss can be written in the form:
\begin{equation}
E[L] = \int (y(x) - h(x))^2p(x)dx + \iint (h(x) - t)^2p(x,t)dtdx
\end{equation}
The second term arises from the intrinsic noise on the data and represents the minimum achievable value of the expected loss. Given a certain dataset $D$, the first term can be decomposed as:
\begin{equation}
E_D[\left\lbrace y(\bf{x}, D)-h(\bf{x}) \right\rbrace^2] + E_D[\left\lbrace y(\bf{x}, D)-E_D[y(\bf{x}, D)] \right\rbrace^2]
\end{equation} where the first term is called bias and the second term is called variance. Our goal is to minimize the expected loss, which we have decomposed into the sum of a (squared) bias, a variance, and a constant noise term.


### 4.1.4 Bayesian Linear Regression

#### 4.1.4.1 Parameter distribution
We choose the prior distribution of $\bf{w}$ as the conjugate Gaussian:
\begin{equation}
	p(\bf{w}) = \cal{N} (\bf{w|m}_0,\bf{S}_0).
\end{equation}
Therefore, the posterior distribution is :
\begin{equation}
	p(\bf{w}) = \cal{N} (\bf{w|m}_0,\bf{S}_0).
\end{equation} where
where
\begin{align}
	\bf{m}_N &= \bf{S}_N (\bf{S}_0^{-1}\bf{m}_0 + \beta \bf{\Phi}^{\intercal} \bf{t}) \\
	\bf{S}_N^{-1} &= \bf{S}_0^{-1} + \beta \bf{\Phi}^{\intercal} \bf{\Phi}.
\end{align}. 
Note that $\bf{w}_{MAP} = \bf{m}_N$.

An interesting phenomenon to note is that the logarithm of posterior distribution of parameters $\ln p(\bf{w|t})$ serves as a new error function for MLE method with a regularization term. And the form of prior $p(\bf{w}|\alpha)$ decides the form of the regularization term.

#### 4.1.4.2 Predictive distribution
We are more interested in making predictions, and the predictive distribution takes the form:
\begin{equation}
	p(t|\bf{x,t},\alpha,\beta) = \int p(t|\bf{w,x},\beta)p(\bf{w}|\bf{t},\alpha,\beta) d \bf{w} = \cal{N}(t|\bf{m}_N^{\intercal} \boldsymbol{\phi}(\bf{x}),\sigma_N^2(\bf{x}))
\end{equation} where the variance $\sigma^2_N(\bf{x})$ of the predictive distribution is given by
\begin{equation}
	\sigma_N^2(\bf{x}) = \frac{1}{\beta} + \boldsymbol{\phi}(\bf{x})^{\intercal} \bf{S}_N \boldsymbol{\phi}(\bf{x})
\end{equation}
The first term in represents the noise on the data whereas the second term
reflects the uncertainty associated with the parameters $\bf{w}$.


If we treat both $\bf{w}$ and $\beta$ as unknown (i.e., both of them are parameters), then we can introduce a conjugate prior distribution $p(\bf{w},\beta)$ given by a Gaussian-gamma distribution. In this case, the predictive distribution is a Student's t-distribution.


#### 4.1.4.3 Bayesian Model Comparison
The Bayesian view of model comparison simply involves the use of probabilities to represent uncertainty in the choice of model, along with a consistent application of the sum and product rules of probability. The posterior distribution of a model is defined by:
\begin{equation}
	p(\cal{M}_i|\cal{D}) \propto p(\cal{M}_i)p(\cal{D}|\cal{M}_i)
\end{equation}
where
\begin{equation}
	p(\cal{D}|\cal{M}_i) = \int p(\cal{D}|\bf{w},\cal{M}_i)p(\bf{w}|\cal{M}_i) d \bf{w}
\end{equation} which is called the model evidence. 



### 4.1.5 The Evidence Approximation
When discussing the Bayesian linear regression, we make predictions by marginalizing with respect to the parameters $\bf{w}$. However, we did not consider the priors on hyperparameters such as $\alpha$ and $\beta$, which makes it not a full Bayesian approach. In the full Bayesian approach, we make predictions also by marginalizing with respect to these hyperparameters. Although we can integrate analytically over either $\bf{w}$ or over the hyperparameters, the complete marginalization over all of these variables is analytically intractable. Here we make an approximation to the full Bayesian approach by setting the hyperparameters to specific values determined by maximizing the \imp{marginal likelihood function} obtained by first integrating over the parameters $\bf{w}$.


This framework is known in the statistics literature as empirical Bayes, or type 2 maximum likelihood, or generalized maximum likelihood and in machine learning literature is also called the evidence approximation.


If we introduce hyperpriors over $\alpha$ and $\beta$, the predictive distribution is obtained by marginalizing over $\bf{w}$, $\alpha$ and $\beta$ so that
\begin{equation}
p(t|\bf{t}) = \iiint  p( t |\bf{w}, \beta)p(\bf{w}|\beta, \alpha)p(\alpha, \beta|\bf{t}) dw d\alpha d\beta
\end{equation}
As we have discussed , it is analytically intractable. If the posterior distribution $p(\alpha, \beta|t)$ is sharply peaked around values $\hat{\alpha}$ and $\hat{\beta}$, then the predictive distribution is obtained simply by marginalizing over $\bf{w}$ in which $\alpha$ and $\beta$ are fixed to the values $\hat{\alpha}$ and $\hat{\beta}$, so that
\begin{equation}
	p(t|\bf{t}) \approx p(t|\bf{t}, \hat{\alpha}, \hat{\beta}) = \int p(t|\bf{w},\hat{\beta}) p(\bf{w}|\hat{\alpha}, \hat{\beta}) d \bf{w}
\end{equation}


Then we move on to calculate the $\hat{\alpha}$ and $\hat{\beta}$. The posterior distribution of $\alpha$ and $\beta$ take the form:
\begin{equation}
p(\alpha, \beta | t) \propto p(t|\alpha, \beta)p(\alpha, \beta)
\end{equation}
If the prior is relatively flat, then in the evidence framework the values of $\alpha$ and $\beta$ are obtained by maximizing the marginal likelihood function $p(t|\alpha, \beta)$.


#### 4.1.5.1 Evaluation of the evidence function
The marginal likelihood function $p(\bf{t}|\alpha,\beta)$ is obtained by integrating over the weight parameters $\bf{w}$, so that
\begin{equation}
	p(\bf{t}|\alpha,\beta) = \int p(\bf{t}|\bf{w},\beta) p(\bf{w}|\alpha) d \bf{w}
\end{equation}
We can evaluate this integral using the formula for the conditional distribution in a linear-Gaussian model. The result is
\begin{equation}
	\ln p(\bf{t}|\alpha,\beta) = \frac{M}{2} \ln \alpha + \frac{N}{2} \ln \beta - E(\bf{m}_N)-\frac{1}{2}\ln |\bf{A}| - \frac{N}{2}\ln(2 \pi)
\end{equation}
where
\begin{align}
	E(\bf{w}) &= \beta 	E_D(\bf{w}) + \alpha E_W(\bf{w}) \notag \\
	&= \frac{\beta}{2} ||\bf{t}-\boldsymbol{\Phi}\bf{w}||^2 + \frac{\alpha}{2} \bf{w}^{\intercal} \bf{w}
\end{align}
and
\begin{equation}
	\bf{A} = \alpha \bf{I} + \beta \bf{\Phi}^{\intercal} \bf{\Phi}.
\end{equation}


#### 4.1.5.2 Maximizing the evidence function
> the calculation remains to be understood

As a result, we get an implicit solution for $\alpha$:
\begin{equation}
	\alpha = \frac{\gamma}{\bf{m}_N^{\intercal} \bf{m}_N}.
\end{equation}
Then we follow the iterative equation solving scheme to get the optimal $\alpha$.
\begin{equation}
	\frac{1}{\beta} = \frac{1}{N-\gamma} \sum_{n=1}^N \{ t_n - \bf{m}_N^{\intercal} \boldsymbol{\phi}(\bf{x}_n) \}^2.
\end{equation}
Again this is an implicit solution for $\beta$.


### 4.1.6 EM for Bayesian linear regression
Our goal is to maximize the evidence function $p(\bf{t}|\alpha,\beta)$ with respect to $\alpha$ and $\beta$. Because the parameter vector $\bf{w}$ is **marginalized out**, we can regard it as a **latent variable**, and hence we can optimize this marginal likelihood function using EM.

* The M step: We have already derived the posterior distribution of $\bf{w}$ and the complete-data log likelihood function is given by
\begin{equation}
	\ln p(\bf{t,w}|\alpha,\beta) = \ln p(\bf{t|w},\beta) + \ln p(\bf{w}|\alpha).
\end{equation}
* Taking the expectation with respect to the posterior distribution of $\bf{w}$ then gives
$$
\begin{align}
	{E}[\ln p(\bf{t,w}|\alpha,\beta)] &= \frac{M}{2} \ln \left( \frac{\alpha}{2\pi} \right) - \frac{\alpha}{2} {E}[\bf{w}^{\intercal} \bf{w}]+\frac{N}{2}\ln \left( \frac{\beta}{2\pi} \right) \notag \\
	&\quad -\frac{\beta}{2}\sum_{n=1}^N {E}[(t_n - \bf{w}^{\intercal} \boldsymbol{\phi}_n)^2].
\end{align}
$$
* Setting the derivatives with respect to $\alpha$ to zero, we obtain the M step re-estimation equation
\begin{equation}
	\alpha = \frac{M}{{E}[\bf{w}^{\intercal} \bf{w}]} = \frac{M}{\bf{m}_N^{\intercal} \bf{m}_N + {Tr}(\bf{S}_N)}.
\end{equation}
An analogous result holds for $\beta$.



### 4.1.7 Variational full Bayesian







## 4.2 Linear Models for Classification
The goal in classification is to take an input vector $\bf{x}$ and to assign it to one of $K$ discrete classes $C_k$ where $k = 1,...,K$. Note that the goal is different from that in the regression problem, which is to fit a function or a distribution. For regression problems, the target variable $t$ is simply the vector of real numbers whose values we wish to predict. For classification problems, the input data is assigned discrete labels. There are three distinct approaches to the classification problem. The simplest involves constructing a discriminant function that directly assigns each vector $\bf{x}$ to a specific class. A more powerful approach, however, models the conditional probability distribution $p(C_k|x)$ in an inference stage, and then subsequently uses this distribution to make optimal decisions.


There are two different approaches to determining the conditional probabilities $p(C_k|x)$. One technique is to model them directly. The other is to adopt a generative approach, then we compute the required posterior probabilities using Bayes’ theorem:
\begin{equation}
p(C_k|\bf{x}) = \frac{p(\bf{x} | C_k)p(C_k)}{p(\bf{x})}
\end{equation}

### 4.2.1 Discriminant functions

#### 4.2.1.1 Least squares for classification
The least-squares solutions lack robustness to outliers, and this applies equally to the classification application. 
![屏幕快照 2019-03-05 下午6.14.45.png](resources/7204FB64706C12F382603C91098B23A7.png =756x431)
However, problems with least squares can be more severe than simply lack of robustness. 
![屏幕快照 2019-03-05 下午6.15.38.png](resources/A46D575D50BAAEC0388EBF69FA2D5194.png =771x439)
The least squares corresponds to maximum likelihood under the assumption of a Gaussian conditional distribution, whereas binary target vectors clearly have a distribution that is far from Gaussian.


#### 4.2.1.2 Fisher's linear discriminant
Fisher's linear discriminant aims at projecting the data to a hyperplane in lower dimension, which at the same time gives a large separation between the projected class means while also giving a small variance within each class, thereby minimizing the class overlap.
\begin{equation}
	\bf{y} = \bf{W}^{\intercal} \bf{x}.
\end{equation}
The fisher criterion takes the form:
\begin{equation}
J(w) = \frac{(m_2 - m_1)^2}{s_1^2 + s_2^2}
\end{equation} which is the ratio of the between-class variance to the within-class variance. 

When $J(w)$ is maximized with respect to $w$, we have:
\begin{equation}
w \propto S_W^{-1}(m_2 - m_1)
\end{equation}
Here I would like to explain the result a little. The example on the book is under the 2 classes. Therefore the  $\bf{S_B} w$ is always in the direction of $(m2 −m1)$, and ignoring the parameter leads to the given result. 
If there are $K$ classes, the within-class and between-class matrices are defined by
$$
\begin{align}
	\bf{S}_W &= \sum_{k=1}^K \sum_{n\in \cal{C}_k} (\bf{y}_n - \boldsymbol{\mu}_k)(\bf{y}_n - \boldsymbol{\mu}_k)^{\intercal} \\
	\bf{S}_B &= \sum_{k=1}^K N_k (\boldsymbol{\mu}_k -\boldsymbol{\mu})(\boldsymbol{\mu}_k-\boldsymbol{\mu})^{\intercal}
\end{align}
$$
where there are $K$ classes and
\begin{equation}
	\boldsymbol{\mu}_k = \frac{1}{N_k} \sum_{n \in \cal{C}_k} \bf{y}_n, \qquad \boldsymbol{\mu} = \frac{1}{N} \sum_{k=1}^{K}N_k \boldsymbol{\mu}_k.
\end{equation}
$$
\begin{equation}
	J(\bf{W}) = {Tr}\{ \bf{s}_W^{-1} \bf{s}_B \}.
\end{equation}
$$

#### 4.2.1.3 The perceptron algorithm
The model is:
\begin{equation}
	y(\bf{x}) = f(\bf{w}^{\intercal} \phi(\bf{x}))
\end{equation}
where the nonlinear activation function $f(\cdot)$ is given by a step function of the form
\begin{equation}
	f(a) = \begin{cases}
		+1, \quad & a \geq 0 \\
		-1, \quad & a < 0.
	\end{cases}
\end{equation}
And the perceptron criterion is
$$
\begin{equation}
	E_P(\bf{w}) = - \sum_{n \in \cal{M}} \bf{w}^{\intercal} \boldsymbol{\phi}_n t_n
\end{equation} 
$$
where ${\phi_n} = \phi(x_n)$ and $\cal{M}$ denotes the mis-classified set. 
Using the stochastic gradient descent, we have:
\begin{equation}
w^{\tau+1} = w^{\tau} - \eta \nabla E_P(w) = w^{\tau} + \eta\phi_nt_n
\end{equation}
The <cite>perceptron convergence theorem</cite> states that if there exists an ex- act solution (in other words, if the training data set is linearly separable), then the perceptron learning algorithm is guaranteed to find an exact solution in a finite number of steps.


### 4.2.2 Probabilistic generative models
Since I am quite familiar with this, I left out much details and only gave the conclusion:


As long as the generative probability $p(\bf{x}|\cal{C}_k)$ belongs to the exponential family, the posterior distribution $p(\cal{C}_k|\bf{x})$ will always have the form of a logistic or softmax function, with linear parameters.


### 4.2.3 Probabilistic discriminative models
Here we focus on the logistic regression model:
\begin{equation}
	p(\cal{C}_1|\boldsymbol{\phi}) = \sigma(\bf{w}^{\intercal} \boldsymbol{\phi})
\end{equation}
with $$p(\cal{C}_2|\boldsymbol{\phi}) = 1-p(\cal{C}_1|\boldsymbol{\phi})$$.


#### 4.2.3.1 Logistic regression
The likelihood function takes the form:
\begin{equation}
p(\bf{t}|w) = \prod_{n=1}^N y_n^{t_n}(1-y_n)^{1-t_n}
\end{equation} and take the logarithm we have:
\begin{equation}
	E(\bf{w}) = -\ln p(\bf{t|w}) = -\sum_{n=1}^N \{ t_n \ln y_n + (1-t_n) \ln (1-y_n) \}
\end{equation} where $y_n = \sigma(\bf{w}^{\intercal}\phi_n)$.
Taking the gradient of the error function with respect to $\bf{w}$, we obtain
\begin{align}
	\nabla E(\bf{w}) &= \sum_{n=1}^N (y_n -t_n) \boldsymbol{\phi}_n = \boldsymbol{\Phi}^{\intercal} (\bf{y-t})\\
	\bf{H} = \nabla\nabla E(\bf{w}) &= \sum_{n=1} ^N  y_n(1-y_n)\boldsymbol{\phi}_n\boldsymbol{\phi}_n^{\intercal} = \boldsymbol{\Phi}^{\intercal}\bf{R} \boldsymbol{\Phi}
\end{align}
We can use gradient descent to solve this problem. 


Note that the MLE method can exhibit severe over-fitting for data sets that are linearly separable. The logistic sigmoid function becomes infinitely steep in feature space, corresponding to a Heaviside step function, when $w$ goes infinite. 


#### 4.2.3.2 Iterative reweighted least squares
The Newton-Raphson update, for minimizing a function $E(\bf{w})$, takes the form:
\begin{equation}
\bf{w}^{new} = \bf{w}^{old} - \boldsymbol{H}^{-1}\nabla E(\bf{w})
\end{equation}
In logistic regression, the formula became:
\begin{equation}
	\bf{w}^{(new)} = (\boldsymbol{\Phi}^{\intercal} \bf{R} \boldsymbol{\Phi})^{-1} \boldsymbol{\Phi}^{\intercal} \bf{Rz}	
\end{equation}
where
\begin{equation}
	\bf{z} = \boldsymbol{\Phi} \bf{w}^{(old)} - \bf{R}^{-1}(\bf{y-t}).
\end{equation} 
And $\bf{R}$ is defined as:
\begin{equation}
R_{nn} = y_n(1-y_n)
\end{equation}

#### 4.2.3.3 Other optimization methods
See chapter 3




### 4.2.4 Probabilistic discriminative models with Bayesian methods
Here we focus on the logistic regression in a Bayesian way.

#### 4.2.4.1 Laplace Approximation
Laplace approximation aims to find a Gaussian approximation to a probability density defined over a set of continuous variables. The center(mean) of this Gaussian distribution is the point $z_0$ where $p'(z_0) = 0$, and the covariance is the Hessian matrix:
\begin{equation}
\boldsymbol{H} = -\nabla \nabla lnf(z)|_{z = z_0}
\end{equation}


Considering logistic regression, the posterior distribution over $\bf{w}$ is given by:
\begin{equation}
	p(\bf{w|t}) \propto p(\bf{w}) p(\bf{t|w})
\end{equation}
\begin{align}
	\ln p(\bf{w|t}) = &-\frac{1}{2} (\bf{w-m}_0)^{\intercal} \bf{S}_0^{-1} (\bf{w-m}_0) \\
	&+ \sum_{n=1}^N \{ t_n \ln y_n + (1-t_n)\ln (1-y_n) \} + {const.}
\end{align}
Here we use Laplace approximation to approximate the posterior distribution:
\begin{equation}
q(w) = \cal{N}(w|\bf{w_{MAP}}, S_N)
\end{equation} where $\bf{S_N}$ takes the form:
\begin{equation}
	\bf{S}_N^{-1} = - \nabla\nabla \ln p(\bf{w}_{MAP}|\bf{t}) = \bf{S}_0^{-1} + \sum_{n=1}^N y_n(1-y_n)\boldsymbol{\phi}_n \boldsymbol{\phi}_n^{\intercal}.
\end{equation}
Similar to the Bayesian linear regression, we would like to take use of this distribution to make predictions. 


#### 4.2.4.2 Predictive distribution
The predictive distribution can be written as:
\begin{equation}
	p(\cal{C}_1|\boldsymbol{\phi},\bf{t})=\int p(\cal{C}_1|\boldsymbol{\phi},\bf{w})p(\bf{w|t}) d \bf{w} \simeq \int \sigma(\bf{w}^{\intercal}\boldsymbol{\phi})q(\bf{w})d \bf{w}
\end{equation}
We finally get
\begin{equation}
	p(\cal{C}_1|\boldsymbol{\phi},\bf{t}) = \sigma(\kappa(\sigma_a^2)\mu_a)
\end{equation}
where
\begin{equation}
	\sigma_a = \boldsymbol{\phi}^{\intercal} \bf{S}_N \boldsymbol{\phi}.
\end{equation}
I gave the detailed deduction when writing the book, which is omitted here. 



#### 4.2.4.3 Variational logistic regression

## 5. Neural Networks
In order to apply linear models mentioned above to large-scale problems, it is necessary to adapt the basis functions to the data. One way is to adjust the number of the basis functions such as SVM，RVM. Another way is to fix the number of basis functions in advance but allow them to be adaptive, in other words to use parametric forms for the basis functions in which the parameter values are adapted during training.

### 5.1 Feed-forward Network Functions
The model takes the form:
\begin{align}
	a_j &= \sum_{i=0} ^D w_{ji}^{(1)} x_i \\
	y_k(\bf{x,w}) &= g\left( \sum_{j=0}^M w_{kj}^{(2)} h(a_j) \right).
\end{align}
where $g$ and $h$ are activation functions such as logistic sigmoid function, relu function, tanh and so on. 
![屏幕快照 2019-03-07 下午3.40.03.png](resources/88F507736C97F2334E73249C74BA3B8D.png =582x317)
Neural networks are therefore said to be universal approximators. For example, a two-layer network with linear outputs can uniformly approximate any continuous function on a compact input domain to arbitrary accu- racy provided the network has a sufficiently large number of hidden units. This result holds for a wide range of hidden unit activation functions, but excluding polynomials. The key problem is how to find the proper parameters. 

### 5.2 Weight-space symmetries
<cite>Chen et al.(1993)</cite> have proved that multiple distinct choices for the weight vector $\bf{w}$ can all give rise to the same mapping function from inputs to outputs. Concisder the two-layer network with $M$ hidden units, there will be $M$ such ‘sign-flip’ symmetries, and thus any given weight vector will be one of a set $2^M$ equivalent weight vectors. 


Similiarly, the order of different parameters in the weight space does not influence the result. For $M$ hidden units, any given weight vector will belong to a set of $M!$ equivalent weight vectors associated with this interchange symmetry, corresponding to the $M!$ different orderings of the hidden units.


Taking the two factors into consideration, the network will therefore have an overall weight-space symmetry factor of $M!2^M$. If the number of layers is more than 2,  the total level of symmetry will be given by the product of such factors. 

### 5.3 Network Training
In summary, there is a natural choice of both output unit activation function and matching error function, according to the type of problem being solved. For regression we use linear outputs and a sum-of-squares error, for (multiple independent) binary classifications we use logistic sigmoid outputs and a cross-entropy error function, and for multiclass classification we use softmax outputs with the corresponding multiclass cross-entropy error function. The detailed deduction is as following. 


For regression problem, the likelihood function is:
\begin{equation}
p(t|\bf{X}, \bf{w}, \beta) = \prod_{n=1}^Np(t_n\cal{|}x_n, \bf{w}, \beta)
\end{equation} where 
\begin{equation}
p(t|x, \bf{w}) = \cal{N}(t|y(w, \bf{w}), \beta^{-1})
\end{equation}
And take the negative logarithm we have:
\begin{equation}
\frac{\beta}{2}\sum_{n=1}^N \left\lbrace y(x, \bf{w}) - t_n\right\rbrace^2 - \frac{N}{2}ln\beta + \frac{N}{2}ln(2\pi)
\end{equation} which is equivalent to the sum=of-squares error. 


For classification problem, first consider the  the case of binary classification, the conditional distribution of targets given inputs is then a Bernoulli distribution of the form:
\begin{equation}
p(t|x, \bf{w}) = y(x, \bf{w})^t \left\lbrace 1- y(x, \bf{w})\right\rbrace^{1-t}
\end{equation}
And the negative log likelihood is:
\begin{equation}
E{\bf{w}} = - \sum_{n=1}^N \left\lbrace t_nln y_n + (1-t_n)ln(1-y_n) \right\rbrace
\end{equation} which is a cross-entropy error function.  <cite>Simard et al. (2003)</cite> found that using the cross-entropy error function instead of the sum-of-squares for a classification problem leads to faster training as well as improved generalization. Then we came to the standard multiclass classification problem in which each input is assigned to one-of-K mutually exclusive classes. The binary target variables $t_k \in \{0, 1\}$ have a 1-of-K coding scheme indicating the class, and the network outputs are interpreted as $y_k(x,{w}) = p(t_k = 1|x)$, leading to the following error function :
\begin{equation}
E(w) = -\sum_{n=1}^N\sum_{k=1}^K t_{nk}ln y_k(x_n, w)
\end{equation}
More specifically, for single input data $x$, the prediction is:
\begin{equation}
p(t|x_j, \bf{w}) = \prod_{i=1}^K y_i(x_j, \bf{w})^{t_i} 
\end{equation} where $\bf{t}$ is in one-of-K representation. 
Therefore, the negative log likelihood is:
\begin{equation}
E(w) = -{ln} (\prod_{j=1}^N p(t|x_j, \bf{w}))
\end{equation}
Note that the $y_k$ takes the form:
\begin{equation}
y_k(x, \bf{w}) = \frac{exp(a_k(x, \bf{w}))}{\sum_j exp(a_j(x, \bf{w}))}
\end{equation}

#### 5.3.1 Parameter optimization
We can use gradient descent to optimize the parameters. 
\begin{equation}
w^{\tau+1} = w^{\tau} + \delta w^{\tau}
\end{equation}

#### 5.3.2 Local quadratic approximation
In the new coordinate system, whose basis vectors are given by the eigenvectors $\{u_i\}$, the contours of constant $E$ are ellipses centred on the origin, as illustrated below. For a one-dimensional weight space, a stationary point $w*$ will be a minimum if
\begin{equation}
\frac{\partial^2 E}{\partial w^2} |_{w*} > 0
\end{equation}

#### 5.3.3 Gradient descent optimization
\begin{equation}
w^{\tau + 1} = w^{\tau} - \eta \nabla E(w^{\tau})
\end{equation}

#### 5.3.4 Error Backpropagation
The backpropagation procedure can be summarized as follows:
* Apply an input vector $x_n$ to the network and forward propagate through the network using
\begin{equation}
a_j = \sum_i w_{ji}z_i
\end{equation} and
\begin{equation}
z_j = h(a_j)
\end{equation} to find the activations of all the hidden and output units.
* Calaulate the $\delta_k$ for all output units:
\begin{equation}
\delta_k = y_k - t_k
\end{equation}
* Backpropagate the $\delta$’s using
\begin{equation}
\delta_j = h'(a_j)\sum_kw_{kj}\delta_k
\end{equation}
to obtain δj for each hidden unit in the
network.
* Use 
\begin{equation}
\frac{\partial E_n}{\partial w_{ji}} = \delta_j z_j
\end{equation}
to evaluate the required derivatives.

I took the artificial neural network class last term and wrote the error backpropagation codes, which could be found in my github repository. 


### 5.4 Regularization in Neural Networks

#### 5.4.1 Consistent Gaussian priors
Any regularizer should be consistent with this property, otherwise it arbitrarily favours one solution over another, equivalent one. We can zoom the parameters to offset the re-scaling of the weights. 

#### 5.4.2 Early stopping
An alternative to regularization as a way of controlling the effective complexity of a network is the procedure of early stopping. The error measured with respect to independent data, generally called a validation set, often shows a decrease at first, followed by an increase as the network starts to overfit. Training can therefore be stopped at the point of smallest error with respect to the validation data set. 

#### 5.4.3 Invariances
In many applications of pattern recognition, it is known that predictions should be unchanged, or invariant, under one or more transformations of the input vari- ables. The approaches to achieve invariance can be divided into four categories. 
> 1. The training set is augmented using replicas of the training patterns, trans- formed according to the desired invariances.
> 2. A regularization term is added to the error function that penalizes changes in the model output when the input is transformed. 
> 3. Invariance is built into the pre-processing by extracting features that are invari- ant under the required transformations.
> 4. Build the invariance properties into the structure of a neural network (or into the definition of a kernel function in the case of techniques


#### 5.4.4 Tangent propagation
Encourage local invariance in the neighbourhood of the data points, by the addition to the original error function $E$ of a regularization function $\Omega$ to give a total error function of the form:
\begin{equation}
\tilde{E} = E + \lambda\Omega
\end{equation} where $\lambda$ is the regularization coefficient and
\begin{equation}
\Omega = \frac{1}{2}\sum_{n}\sum_k (\sum_{i=1}^DJ_{nki}\tau_{ni})^2
\end{equation}



#### 5.4.5 Training with transformed data

#### 5.4.6 Convolutional neural networks
These notions are incorporated into convolutional neural networks through three mechanisms: (i) local receptive fields, (ii) weight sharing, and (iii) subsampling.
![屏幕快照 2019-03-07 下午10.59.32.png](resources/3CCDFDD295DC50F1A8C70E7152E73B91.png =647x395)
By the way, I have wrote the cnn codes which could be found in my github repository. After coding, I found that a convolutional layer was just a sparse fully-connected layer, which used fewer parameters to learn a better representation. Convolutional layer is an efficient structure itself and it can help to avoid overfitting. 


#### 5.4.7 Soft weight sharing
In fact, we have seen weight sharing in convolutional neural networks, which build translation invariance into networks. But we can loosen the restriction and here comes the soft weight sharing. In this approach, the hard constraint of equal weights is replaced by a form of regularization in which groups of weights are encouraged to have similar values.


Consider the prior over parameters $\bf{w}$ to be a mixture of Gaussians. The centers, variances and mixing coefficients will be considered as adjustable parameters to be determined as part of the learning process. Thus we have
\begin{equation}
	p(\bf{w}) = \prod_i p(w_i)
\end{equation}
where
\begin{equation}
	p(w_i) = \sum_{j=1}^M \pi_j \cal{N}(w_j|\mu_j,\sigma_j^2).
\end{equation}
Thus the refularization term takes the form:
\begin{equation}
\Omega(\bf{w}) = - \ln p(\bf{w})
\end{equation}
And the error function is:
\begin{equation}
\tilde{E}(\bf{w}) = E(\bf{w}) + \lambda \Omega(\bf{w})
\end{equation}
The posterior distribution over $\pi_j$ is:
\begin{equation}
\gamma_j(w) = p(\pi_j|w) = \frac{\pi_j\cal{N}(w|\mu_j, \sigma_j^2)}{\sum_k\pi_k\cal{N}(w|\mu_k, \sigma_k^2)}
\end{equation}
This formula uses Bayes' theorem:
\begin{equation}
p(a|b) = \frac{p(b|a)p(a)}{p(b)}
\end{equation} where $p(a) = \pi_j$, $p(b|a) = \cal{N}(w|\mu_j, \sigma_j^2)$. 
The derivatives of the total error function with respect to the weights are then given by:
\begin{equation}
\frac{\partial\tilde{E}}{\partial w_i} = \frac{\partial E}{\partial w_i} + \lambda \sum_j \gamma_j(w_i)\frac{w_i - \mu_j}{\sigma_j^2}
\end{equation}
And this can be be interpreted that the effect of the regularization term is to pull each weight towards the centre of the jth Gaussian, with a force proportional to the posterior probability of that Gaussian for the given weight. The derivatives with respect to the other parameters can be calculated in the same manner. We can use standard optimization algorithm such as conjugate gradients or qusi-Newton methods to solve the problem. 



### 5.6 Mixture density networks
For any given value of $x$, the mixture model provides a general formalism for modelling an arbitrary conditional density function $p(t|x)$. The mixing coefficients as well as the component densities are flexible functions of the input vector $x$. 


Here we only consider the components to be Gaussian, so that:
\begin{equation}
p(t|x) = \sum_{k=1}^K \pi_k(x) \cal{N}(t|\mu_k(x), \sigma_k^2(x)\cal{I})
\end{equation} which is an example of heteroscedastic model since the noise variance on the data is a function of the input vector $\bf{x}$. 
The negative log likelihood is given by
\begin{equation}
	E(\bf{w})= - \sum_{n=1}^N \ln \left\{ \sum_{k=1}^k \pi_k(\bf{x}_n,\bf{w})\cal{N}(\bf{t}_n|\boldsymbol{\mu}_k(\bf{x}_n,\bf{w}),\sigma_k^2(\bf{x}_n,\bf{w})) \right\}
\end{equation} where $\sigma_k(x) = \exp(a_k^{\sigma})$ and $\mu_{kj}(x) = a_{kj}^{\mu}$. We use the network's output with certain activation to be the mixture density network's parameters. We need to calculate the derivatives of the error $E(\bf{w})$ with respect to the components of $w$. These can be evaluated by using the standard backpropagation procedure. The derivatives with respect to the network output activations governing the mixing coefficients are given by
\begin{align}
	\frac{\partial E_n}{\partial a_{k}^{\pi}} &= \pi_k - \gamma_{nk} \\
	\frac{\partial E_n}{\partial a_{kl}^{\mu}} &= \gamma_{nk} \left\{ \frac{\mu_{kl}-t_{nl}}{\sigma_k^2} \right\} \\
	\frac{\partial E_n}{\partial a_{k}^{\sigma}} &= \gamma_{nk} \left\{ L - \frac{||\bf{t}_n-\boldsymbol{\mu}_k||^2}{\sigma_k^2} \right\}
\end{align}


### 5.7 Bayesian neural networks
In a Bayesian treatment we need to marginalize over the distribution of parameters in order to make predictions just like what we did in Bayesian logistic regression. 

#### 5.7.1 Regression
Similarto the Bayesian logistic regression, we have: 
\begin{align}
	p(\cal{D}|\bf{w},\beta) &= \prod_{n=1}^N \cal{N}(t_n|y(\bf{x}_n,\bf{w}),\beta^{-1}) \\
	p(\bf{w}|\cal{D},\alpha,\beta) &\propto p(\bf{w}|\alpha) p(\cal{D}|\bf{w},\beta) \\
	\ln p(\bf{w}|\cal{D}) &= -\frac{\alpha}{2} \bf{w}^{\intercal} \bf{w} - \frac{\beta}{2} \sum_{n=1}^N \{ y(\bf{x}_n,\bf{w})-t_n \}^2 + {const}.
\end{align} 
The posterior distibution will be non-Gaussian as a consequence of the nonlinear dependence of $y(x, \bf{w})$ on $\bf{w}$. 


We can find a Gaussian approximation to the posterior distribution by using the Laplace approximation. Therefore, we have to find a (local) maximum of the posterior, and this must be done using iterative numerical optimization. 
\begin{equation}
\ln p(\bf{w}|\cal{D}) = -\frac{\alpha}{2} \bf{w}^{\intercal} \bf{w} - \frac{\beta}{2} \sum_{n=1}^N \{ y(\bf{x}_n,\bf{w})-t_n \}^2 + {const}
\end{equation}
Suppose we have found the maximum point $\bf{w}_{MAP}$, we can then build a local Gaussian approximation by evaluating the matrix of second derivatives of the negative log posterior distribution, which is given by
\begin{equation}
	\bf{A} = -\nabla\nabla \ln p(\bf{w}_{MAP}|\cal{D},\alpha,\beta) = \alpha \bf{I} + \beta \bf{H}
\end{equation} where $\bf{H}$ is the Hessian matrix comprising the second derivatives of the sum-of-squares error function with respect to the components of $\bf{w}$. We then get the corresponding Gaussian approximation to the posterior
\begin{equation}
	q(\bf{w}|\cal{D}) = \cal{N}(\bf{w}|\bf{w}_{MAP},\bf{A}^{-1}).
\end{equation}

Similarly, the predictive distribution is obtained by marginalizing with respect to this posterior distribution
\begin{equation}
	p(t|\bf{x},\cal{D}) = \int p(t|\bf{x,w})q(\bf{w}|\cal{D}) d \bf{w}.
\end{equation}
We assume that the posterior distribution has small variance compared with the characteristic scales of $\bf{w}$ over which $y(\bf{x,w})$ is varying. This allows us to make a Taylor expansion of the network function around $\bf{w}_{MAP}$ and retain only the linear terms
\begin{equation}
	y(\bf{x,w}) \simeq y(\bf{x},\bf{w}_{MAP}) + \bf{g}^{\intercal} (\bf{w}-\bf{w}_{MAP}).
\end{equation}
so that
\begin{equation}
	p(t|\bf{x,w},\beta) \simeq \cal{N}(t|y(\bf{x,w}_{MAP})+\bf{g}^{\intercal} (\bf{w-w}_{MAP}),\beta^{-1}).
\end{equation}
We can therefore use the formula for **linear-Gaussian** model to obtain
\begin{equation}
	p(t|\bf{x},\cal{D},\alpha,\beta) = \cal{N}(t|y(\bf{x},\bf{w}_{MAP}),\sigma^2(\bf{x}))
\end{equation}
where the input-dependent variance is given by
\begin{equation}
	\sigma^2(\bf{x}) = \beta^{-1} + \bf{g}^{\intercal} \bf{A}^{-1}\bf{g}.
\end{equation}
The variance has two terms, the first of which arises from the intrinsic noise on the target variable, whereas the second is an $x$-dependent term that expresses the uncertainty in the interpolant due to the uncertainty in the model parameters $w$.


#### 5.7.1.1 Hyperparameter optimization
The evidence for the hyperparameters is obtained by
\begin{equation}
	p(\cal{D}|\alpha,\beta) = \int p(\cal{D}|\bf{w},\beta)p(\bf{w}|\alpha) d \bf{w}.
\end{equation}
Others remains to be understood. 



### 5.7.2 Classification
As for the classification problem, the log likelihood function for this model is given by:
\begin{equation}
\ln p(D|w) = \sum_{n=1}^N \{t_n \ln y_n + (1-t_n)\ln(1-y_n)\}
\end{equation} where $t_n \in \{0,1\}$ is the target and $y_n = y(x_n, \bf{w})$. Note that there is no hyperparameter $\beta$, because the data points are assumed to be correctly labelled. As before, the prior is taken to be an isotropic Gaussian of the form $p(w|\alpha) = \cal{N}(w|0, \alpha^{-1}I)$. 


The first stage in applying the Laplace framework to this model is to initialize the hyperparameter $\alpha$, and then to determine the parameter vector $w$ by maximizing the log posterior distribution. This is equivalent to minimizing the regularized error function:
\begin{equation}
E(w) = -\ln p(D|w) + \frac{\alpha}{2}w^{\intercal}w
\end{equation} And we can adopt the backpropagation combined with standard optimization algorithms to solve this problem. Having found a solution $w_{MAP}$ for the weight vector, the next step is to evaluate the Hessian matrix $\bf{H}$ comprising the second derivatives of the negative log likelihood function. And the posterior distribution over $w$ takes the form: $\cal{N}(w_{MAP}, A^{-1})$, as seen in the regression problem. 


Then we can maximize the marginal likelihood
\begin{equation}
\ln p(D|\alpha) \approx -E(w_{MAP}) - \frac{1}{2}\ln|A| + \frac{W}{2}\ln(\alpha)
\end{equation}
where $E(w_{MAP})$ has the form:
\begin{equation}
E(w_{MAP}) = -\sum_{n=1}^N \{t_N \ln y_n + (1-y_n)\ln (1-y_n)\} + \frac{\alpha}{2}w_{MAP}^{\intercal}w_{MAP}
\end{equation}
After quite a lot mathematical manipulations, the prediction distribution is:
\begin{equation}
p(t=1|x, D) = \sigma(\kappa(\sigma_{\alpha}^2)a_{MAP})
\end{equation}
The result is similar to the Bayesian logistic regression. 

## 6. Kernel Methods
For the models we have discussed above, the training data is discarded after learning a suitable $\bf{w}$. However, there exists a class of pattern recognition techniques, in which the training data points, or a subset of them, are kept and used also during the prediction phase. 


For models based on a fixed nonlinear feature space mapping $\phi(x)$, the kernel function is given by the relation:
\begin{equation}
k(x, x') = \phi(x)^{\intercal}\phi(x')
\end{equation}
There are several kinds of kernel functions:
> [1] the linear kernel: $k(x, x') = x^{\intercal}x'$
> [2] the stationary kernel: $k(x, x') = k(x - x')$, invariant to translations in input space
> [3] the homogeneous kernel: $k(x, x') = k(|x-x'|)$, also called radial basis function because it depends only on the magnitude of the distance (typically Euclidean) between the arguments


The general idea of kernel substitution is that, if we have an algorithm formulated in such a way that the input vector $x$ enters only in the form of scalar products, then we can replace that scalar product with some other choice of kernel. 

### 6.1 Dual Representations
Consider a linear model whose parameters are determined by minimizing a regularized sum-of-squares error function given by:
\begin{equation}
J(w) = \frac{1}{2}\sum_{n=1}^N\{w^{\intercal}\phi(x_n) - t_n\}^2 + \frac{\lambda}{2}w^{\intercal}w
\end{equation}
If we set the gradient of $J(w)$ with respect to $w$ equal to zero, we have
\begin{equation}
w = \Phi^{\intercal}\bf{a}
\end{equation} where $\Phi = (\phi(x_1)^T, \phi(x_2)^T, ..., \phi(x_n)^T)$ and $\bf{a} = (a_1, a_2, ..., a_N)^{\intercal}$ and
\begin{equation}
a_n = - \frac{1}{\lambda}\{w^{\intercal}\phi(x_n) - t_n \}
\end{equation}
Combining these two equations leads to
\begin{equation}
\bf{a} = (K + \lambda I_N)^{-1}t
\end{equation} where $K$ is the Gram matrix of the form $K = \Phi \Phi^{\intercal}$. 
\begin{equation}
K_{nm} = \phi(x_n)^{\intercal}\phi(x_m) = k(x_n, x_m)
\end{equation}
If we remove the $w$ in the error function by $\bf{a}$, we have:
\begin{equation}
J(\bf{a}) = \frac{1}{2}\bf{a}^{\intercal}\Phi \Phi^{\intercal}\Phi \Phi^{\intercal} \bf{a} - \bf{a}^{\intercal}\Phi \Phi^{\intercal}t + \frac{1}{2}t^{\intercal}t + \frac{t}{2}a^{\intercal}\Phi\Phi^{\intercal}a
\end{equation}
Since we get the formula $\bf{a} = (K + \lambda I_N)^{-1}t$ by setting the gradient of $J(w)$ with respect to $w$ equal to zero, this formula is the answer of the new error function $J(a)$. If we substitute this back into the linear regression model, we obtain the following prediction for a new input $x$:
\begin{equation}
y(\bf{x}) = w^{\intercal}\phi(\bf{x}) = \bf{a}^{\intercal}\Phi\phi(x) = \bf{k}(\bf{x})^{\intercal}(K + \lambda I_N)^{-1}t
\end{equation} where $\bf{k}(x) = (k(x_1, x), k(x_2, x), ..., k(x_n, x))$.


In the dual formulation, we determine the parameter vector a by inverting an N × N matrix, whereas in the original parameter space formulation we had to invert an M × M matrix in order to determine $w$. The advantage of the dual formulation is that it is expressed entirely in terms of the kernel function $k(x,x')$. We can therefore work directly in terms of kernels and avoid the explicit introduction of the feature vector $\phi(x)$, which allows us implicitly to use feature spaces of high, even infinite, dimensionality.


### 6.2 Constructing Kernels
One approach is to choose a feature space mapping $\phi(x)$ and then use this to find the corresponding kernel. Here the kernel function is defined for a one-dimensional input space by
\begin{equation}
k(x, x') = \phi(x)^{\intercal}\phi(x') = \sum_{i=1}^M\phi_i(x)\phi_i(x')
\end{equation}
where $\phi_i(x)$ are the basis functions.


Another way is to construct kernel functions directly, which is difficult. An alternative way is to build them out of simpler kernels as building blocks. This can be done using the following properties, given valid kernels $k_1(\bf{x,x'})$ and $k_2(\bf{x,x'})$:
\begin{align}
	k(\bf{x,x'}) &= ck_1(\bf{x,x'}) \\
	k(\bf{x,x'}) &= f(\bf{x})k_1(\bf{x,x'})f(\bf{x'})\\
	k(\bf{x,x'}) &= q(k_1(\bf{x,x'}))\\
	k(\bf{x,x'}) &= \exp (k_1(\bf{x,x'}))\\
	k(\bf{x,x'}) &=k_1(\bf{x,x'})+k_2(\bf{x,x'})\\
	k(\bf{x,x'}) &=k_1(\bf{x,x'})k_2(\bf{x,x'})\\
	k(\bf{x,x'}) &=k_3(\boldsymbol{\phi}(\bf{x}),\boldsymbol{\phi}(\bf{x'}))\\
	k(\bf{x,x'}) &=\bf{x}^{\intercal} \bf{Ax'}\\
	k(\bf{x,x'}) &=k_a (\bf{x}_a,\bf{x}_a')+k_b(\bf{x}_b,\bf{x}_b')\\
	k(\bf{x,x'}) &=k_a (\bf{x}_a,\bf{x}_a')k_b(\bf{x}_b,\bf{x}_b')
\end{align}
where $c > 0$ is a constant, $f(\cdot)$ is any function, $q(\cdot)$ is a polynomial with nonnegative coefficients, $\boldsymbol{\phi}(\bf{x})$ is a function from $\bf{x}$ to ${R}^M$, $k_3(\cdot,\cdot)$ is a valid kernel in ${R}^M$, $\bf{A}$ is a symmetric positive semidefinite matrix, $\bf{x}_a$ and $\bf{x}_b$ are variables (not necessarily disjoint) with $\bf{x} = (\bf{x}_a, \bf{x}_b)$, and $k_a$ and $k_b$ are valid kernel functions over their respective spaces. We can use this to prove the following kernels are valid:
> Gaussian kernel: $k(x, x') = \exp(-\frac{|x-x'|^2}{2\sigma^2})$
> k(A_1, A_2) = 2^{|A_1 and A_2|}


Another interesting way is to build kernels from a probabilistic generative model (<cite>Haussler, 1999</cite>). Given a generative model p(x) we can define a kernel by
\begin{equation}
k(x, x') = p(x)p(x')
\end{equation} which shows that if the two input vectors both have high probability, they are similar. 
\begin{align}
k(x,x') &= \sum_i p(x|i)p(x'|i)p(i) \\
k(x,x') &= \int p(x|z)p(x'|z)p(z)dz
\end{align} where $z$ is a continuous latent variable.


Another way to build kernels from generative model is known as the Fisher kernel <cite>Jaakkola and Haussler, 1999</cite>. Consider a parametric generative model $p(x|\theta)$ where $\theta$ denotes the vector of parameters. The Fisher score is:
\begin{equation}
g(\theta, x) = \nabla_{\theta}\ln p(x|\theta)
\end{equation}
The Fisher kernel is defined as
\begin{equation}
k(x, x') = g(\theta, x)^{\intercal}F^{-1}g(\theta, x')
\end{equation} where $F$ is the Fisher information matrix, given by
\begin{equation}
F = E_x[g(\theta, x)g(\theta, x)^{\intercal}
\end{equation}
A final example of a kernel function is the sigmoidal kernel given by
\begin{equation}
k(x, x') = tanh(ax^{\intercal}x' + b)
\end{equation}


As we shall see, in the limit of an infinite number of basis functions, a Bayesian neural network with an appropriate prior reduces to a Gaussian process.


### 6.3 Radial Basis Function Networks
Omitted. 

### 6.4 Gaussian Processes
The predictions of linear regression can be reformulated as
\begin{equation}
	\bf{y } = \boldsymbol{\Phi} \bf{w}
\end{equation}
where $\boldsymbol{\Phi}$ is the design matrix. Since $\bf{w}$ has a Gaussian prior $p(\bf{w}) = \cal{N}(\bf{w}|\bf{0},\alpha^{-1}\bf{I})$, $\bf{y}$ is itself Gaussian, whose mean and covariance are given by
$$
\begin{align}
	{E}[\bf{y}] &= \boldsymbol{\Phi} {E}[\bf{w}] = 0\\
	{cov}[\bf{y}] &= {E}[\bf{y}\bf{y}^{\intercal}] = \boldsymbol{\Phi}{E}[\bf{w}\bf{w}^{\intercal}]\boldsymbol{\Phi}^{\intercal} = \frac{1}{\alpha}\boldsymbol{\Phi}\boldsymbol{\Phi}^{\intercal} = \bf{K}.
\end{align}
$$
where $K$ is the Gram matrix with elements
\begin{equation}
	K_{nm} = k(\bf{x}_n,\bf{x}_m) = \frac{1}{\alpha} \boldsymbol{\phi}(\bf{x}_n)^{\intercal} \boldsymbol{\phi}(\bf{x}_m).
\end{equation}

The fact that $\{ y_n \}$ forms a Gaussian processes means we can choose the mean and covariance to completely specify the process. By symmetry we choose ${E}[\bf{y}] = \bf{0}$. The covariance is chosen to be the Gram matrix of some user-specified kernels, i.e. ${E}[y(\bf{x}_n) y(\bf{x}_m)]=k(\bf{x}_n,\bf{x}_m)$, such as the exponential kernel leading to the Ornstein-Uhlenbeck process.

#### 6.4.1 Linear regression
The model is given by
\begin{equation}
	p(\bf{t|y}) = \cal{N}(\bf{t|y},\beta^{-1} \bf{I}_N)
\end{equation}
where $\bf{y}$ serves as an latent variable and its prior is given by
\begin{equation}
	p(\bf{y}) = \cal{N}(\bf{y|0,K}).
\end{equation}
The kernel function that determines $K$ is typically chosen to express the property that, for points $x_n$ and $x_m$ that are similar, the corresponding values $y(x_n)$ and $y(x_m)$ will be more strongly correlated than for dissimilar points.


The marginal distribution $p(\bf{t})$ is obtained by integrating out the latent variable $\bf{y}$:
\begin{equation}
	p(\bf{t}) = \int p(\bf{t|y})p(\bf{y}) d \bf{y} = \cal{N}(\bf{t}|\bf{0,C})
\end{equation}
where
\begin{equation}
	C(\bf{x}_n,\bf{x}_m) = k(\bf{x}_n,\bf{x}_m) + \beta^{-1} \delta_{nm}. 
\end{equation}

We first derive the joint distribution $p(\bf{t}_{N+1})$:
\begin{equation}
	p(\bf{t}_{N+1}) = \cal{N}(\bf{t}_{N+1}|\bf{0,C}_{N+1})
\end{equation}
where
$$
\begin{equation}
	\bf{C}_{N+1} = \begin{pmatrix}
	 \bf{C}_N & \bf{k} \\
	 \bf{k}^{\intercal} & c
	\end{pmatrix}.
\end{equation}
$$
where $\bf{C}_N$ is the covariance matrix with elements given by (\ref{GauPro1}). For $n,m=1,\cdots,N$, the vector $\bf{k}$ has elements $k(\bf{x}_n,\bf{x}_{N+1})$ for $n=1,\cdots,N$, and the scalar $c=k(\bf{x}_{N+1},\bf{x}_{N+1})+\beta^{-1}$. As a result, the conditional distribution $p(t_{N+1}|\bf{t}_N)$ can be derived readily, given by
\begin{align}
	m(\bf{x}_{N+1}) &= \bf{k}^{\intercal} \bf{C}_N^{-1} \bf{t} \\
	\sigma^2(\bf{x}_{N+1}) &= c - \bf{k}^{\intercal} \bf{C}_N^{-1} \bf{k}
\end{align}
These are the key results that define Gaussian process regression.  In general, a Gaussian process is defined as a probability distribution over functions $y(x)$ such that the set of values of $y(x)$ evaluated at an arbitrary set of points $x_1, . . . , x_N$ jointly have a Gaussian distribution.


Here I would like to briefly summarize the Gaussian process. 
*  [1] The parameter $w$ has the prior $p(w) = \cal{N}(w|0, \alpha^{-1}I)$
*  [2] Take the sequence of $y(x_i)$ as a random variable, which gives $\bf{y} = [y(x_1), y(x_2), ..., y(x_n)]$ and $\bf{y} = \Phi w$. Note that $\bf{y}$ is a linear combination of Gaussian distributed variables given by the elements of $w$ and hence is itself Gaussian. Therefore , we know that **$\bf{y}$ is a Gaussian**. 
*  [3] Find the mean and covariance of $\bf{y}$, which gives: $\cal{N}(\bf{y}|0, K)$. $K$ is the Gram matrix related to the **kernel** $k(x,x')$.
*  [4] In regression, we first have the suppose $p(t_n|y_n) = \cal{N}(t_n|y_n, \beta^{-1})$ and $p(\bf{y}) = \cal{N}(\bf{y}|0, K)$. Then we can calculate the marginal distribution of $\bf{t}$ as: $	p(\bf{t}) = \int p(\bf{t|y})p(\bf{y}) d \bf{y} = \cal{N}(\bf{t}|\bf{0,C})$ and $\bf{C}$ is closely related to the kernel $k(x,x')$. 
*  [5] Then we have $p(\bf{t}_{N+1}) = \cal{N}(\bf{t}_{N+1}|\bf{0,C}_{N+1})$
*  [6] Finally we have $$p(t_{N+1}|\bf{t}_N) = \cal{N}(\bf{k}^{\intercal} \bf{C}_N^{-1} \bf{t}, c - \bf{k}^{\intercal} \bf{C}_N^{-1} \bf{k})$$
which is the key point of Gaussian process regression. 

An advantage of a Gaussian processes viewpoint is that we can consider covariance functions that can only be expressed in terms of an **infinite** number of basis functions.

#### 6.4.2 Hyperparameters
The kernels may have parameters, for example, to control the length scale of the correlations. As a result, these parameters belong to the model hyperparameters. The simplest approach is to make a point estimate of ${\theta}$ by maximizing the log likelihood function
\begin{equation}
 \ln p(\bf{t}|{\theta}) = -\frac{1}{2} \ln |\bf{C}_N| - \frac{1}{2} \bf{t}^{\intercal} \bf{C}_N^{-1} \bf{t} - \frac{N}{2} \ln (2\pi).
\end{equation}
and the gradient with respect to the parameters are
\begin{equation}
	\frac{\partial }{\partial \theta_i} \ln p(\bf{t}|{\theta}) = -\frac{1}{2}{Tr}\left( \bf{C}_N^{-1} \frac{\partial \bf{C}_N}{\partial \theta_i} \right) + \frac{1}{2} \bf{t}^{\intercal} \bf{C}_N^{-1} \frac{\partial \bf{C}_N}{\partial \theta_i}\bf{C}_N^{-1} \bf{t}.
\end{equation}

#### 6.4.3 Automatic relevance determination
With parameters corresponding to different terms in the kernel function, we can determine their relative importance. This leads to the technique of automatic relevance determination (ARD), where we **incorporate a separate parameter for each dimensionality of the input variables and then infer these parameters from the data**.


The ARD framework is easily incorporated into the exponential-quadratic kernel to give the following form of kernel function
\begin{equation}
k(x_n, x_m) = \theta_0 exp\{-\frac{1}{2}\sum_{i=1}^D\eta_i(x_{ni} - x_{mi})^2 \} + \theta_2 + \theta_3\sum_{i=1}^Dx_{ni}x_{mi}
\end{equation}


#### 6.4.4 Linear classification
We first introduce the latent variable $\bf{a}_{N+1}$, which denotes the mean($\sigma(\bf{a})$) of Bernoulli distributions for every point:
\begin{equation}
	p(\bf{a}_{N+1}) = \cal{N}(\bf{a}_{N+1}|\bf{0,C}_{N+1})
\end{equation}
where the covariance matrix $\bf{C}_{N+1}$ has elements given by
\begin{equation}
	C(\bf{x}_n,\bf{x}_m) = k(\bf{x}_n,\bf{x}_m) + \nu \delta_{nm}.
\end{equation}
Here the extra term $\nu \delta_{nm}$ is introduced to {ensure the validity of covariance}. Unlike the regression case, the covariance matrix no longer includes a noise term because we assume that all of the training data points are correctly labelled.


The predictive distribution is given by
\begin{align}
	p(t_{N+1} = 1|\bf{t}_N) &= \int p(t_{N+1} = 1|a_{N+1})p(a_{N+1}|\bf{t}_N) d a_{N+1}\\
	&=  \int \sigma(a_{N+1}) p(a_{N+1}|\bf{t}_N) d a_{N+1}
\end{align}
Since this integral is analytical intractable, there are four approaches to obtain an approximate solution:
* Sampling methods.
* Laplace approximation.
* Variational inference.
* Expectation propagation. Since the true posterior is unimodal.


#### 6.4.5 Laplace approximation
To calculate the prediction distribution, we now focus on the posterior distribution over $a_{N+1}$. 
\begin{equation}
	p(a_{N+1}|\bf{t}_N) = \int p(a_{N+1}|\bf{a}_N)p(\bf{a}_N|\bf{t}_N) d \bf{a}_N
\end{equation}
The conditional distribution $p(a_{N+1}|\bf{a}_N)$ is easy to obtain(similar to the $p(t_{N+1}|t_N)$), given by
\begin{equation}
	p(a_{N+1}|\bf{a}_N) = \cal{N}(a_{N+1}|\bf{k}^{\intercal} \bf{C}_N^{-1}\bf{a}_N, c-\bf{k}^{\intercal}\bf{C}_N^{-1}\bf{k}).
\end{equation}


The posterior distribution $p(a_N|t_N)$ is evaluated using Bayesian formula. First, we know that the prior $p(\bf{a}_{N+1})$ is a Gaussian. Second, we have
\begin{equation}
	p(\bf{t}_N|\bf{a}_N) = \prod_{n=1}^N \sigma(a_n)^{t_n} (1-\sigma(a_n))^{1-t_n} = \prod_{n=1}^N e^{a_n t_n}\sigma(-a_n).
\end{equation}
Then we get the log posterior distribution:
\begin{align}
	\Psi(\bf{a}_N) &= \ln p(\bf{a}_N) + \ln p(\bf{t}_N|\bf{a}_N) \\
	&= -\frac{1}{2} \bf{a}_N^{\intercal} \bf{C}_N^{-1} \bf{a}_N - \frac{N}{2} \ln (2\pi) - \frac{1}{2} \ln |\bf{C}_N| + \bf{t}_N^{\intercal}\bf{a}_N \\
	&\quad  -\sum_{n=1}^N \ln (1+e^{a_n}).
\end{align}


We will use Laplace approximation to approximate this distribution. We need to find the mode of the posterior distribution, and this requires the gradient and Hessian of $\Psi(\bf{a}_N)$:
\begin{align}
	\nabla \Psi(\bf{a}_N) &= \bf{t}_N - \boldsymbol{\sigma}_N - \bf{C}_N^{-1} \bf{a}_N \\
    \bf{H} &= -\nabla\nabla \Psi(\bf{a}_N) =  \bf{W}_N + \bf{C}_N^{-1}
\end{align}
where $\bf{W}_N$ is a diagonal matrix with elements $\sigma(a_n)(1-\sigma(a_n))$. Then we can employ the Newton-Raphson formula to get the optimal parameters $\bf{a}_N$ and its corresponding Hessian. The Gaussian approximation to the posterior distribution
\begin{equation}
	q(\bf{a}_N) = \cal{N}(\bf{a}_N|\bf{a}_N^*,\bf{H}^{-1})
\end{equation}

Afterwards, we know that $p(\bf{a}_N|\bf{t}_N)$ is a Gaussian distribution with the parameters given by
\begin{align}
	{E}[a_{N+1}|\bf{t}_N] &= \bf{k}^{\intercal} (\bf{t}_N - \boldsymbol{\sigma}_N) \\
	{var}[a_{N+1}|\bf{t}_N] &= c - \bf{k}^{\intercal} (\bf{W}_N^{-1}+\bf{C}_N)^{-1} \bf{k}.
\end{align}


Finally, we can approximate the predictive distribution similar to that in Bayesian logistic regression. 

ps：The kernel function is determined by $\theta$, which is learned by the training data(maximize the likelihood $p(t_N|\theta)$). 


#### 6.4.6 Connection to neural networks
In a Bayesian neural network, the prior distribution over the parameter vector $\bf{w}$, in conjunction with the network function $f(\bf{x,w})$, produces a prior distribution over functions from $y(\bf{x})$ where $\bf{y}$ is the vector of network outputs. Neal (1996) has shown that, for a broad class of prior distributions over $\bf{w}$, the distribution of functions generated by a neural network will tend to a Gaussian process in the limit $M \rightarrow \infty$.

## 7. Sparse Kernel Machines
One of the significant limitations of many such algorithms is that the kernel function $k(x_n,x_m)$ must be evaluated for all possible pairs $x_n$ and $x_m$ of training points, which can be computationally infeasible during training and can lead to excessive computation times when making predictions for new data points. Here we shall look at kernel-based algorithms that have **sparse solutions**, so that predictions for new inputs depend only on the kernel function evaluated at a subset of the training data points.

### 7.1 SVM
The SVM is a decision machine and so does not provide posterior probabilities. 

#### 7.1.1 Classification
The support vector machine approaches this problem through the concept of the margin, which is defined to be the smallest distance between the decision boundary and any of the samples. 
![屏幕快照 2019-03-09 上午10.45.17.png](resources/AD6B3C02B38EC325B9DF7E1FED74F6DC.png =726x327)
In support vector machines the decision boundary is chosen to be the one for which the margin is maximized. In the limit $\sigma^2$ → 0, the optimal hyperplane is shown to be the one having maximum margin. The intuition behind this result is that as $\sigma^2$ is reduced, the hyperplane is increasingly dominated by nearby data points relative to more distant ones. In the limit, the hyperplane becomes independent of data points that are not support vectors.


The hyperplane is defined by
\begin{equation}
	y(\bf{w}) = \bf{w}^{\intercal} \boldsymbol{\phi}(\bf{x}) + b.
\end{equation}

After introducing these slack variables, the constraints are replaced by
\begin{equation}
	t_n y(\bf{x}_n) \geq 1 -\xi_n \qquad n = 1,\cdots,N
\end{equation}
in which the slack variables are constrained to satisfy $\xi_n \geq 0$.

We omit some intuitions from geometry. Our goal is now to maximize the margin while softly penalizing points that lie on the wrong side of the margin boundary. We therefore minimize
\begin{equation}
	C\sum_{n=1}^N \xi_n + \frac{1}{2} ||\bf{w}||^2
\end{equation}
where the parameter $C > 0$ controls the trade-off between the slack variable penalty and the margin.


Employ the Lagrange multipliers to give the Lagrangian
\begin{equation}
	L(\bf{w},b,\bf{a},\boldsymbol{\xi},\boldsymbol{\mu}) = \frac{1}{2} ||\bf{w}||^2 + C \sum_{n=1}^N \xi_n - \sum_{n=1}^N a_n \{ t_n y(\bf{x}_n) -1 + \xi_n \} - \sum_{n=1}^N \mu_n \xi_n.
\end{equation}
The corresponding set of KKT conditions are given by
\begin{align}
	a_n &\geq 0 \\
	t_n y(\bf{x}_n) &-1 + \xi_n \geq 0 \\
	a_n(t_n y(\bf{x}_n) &-1+\xi_n) = 0 \\
	\mu_n &\geq 0 \\
	\xi_n & \geq 0 \\
	\mu_n \xi_n &= 0
\end{align}
when $n = 1,\cdots,N$.


We now optimize out $\bf{w}$, $b$ and $\{\xi_n\}$ making use of the definition of $y(\bf{x})$ to give
\begin{align}
	\frac{\partial L}{\partial \bf{w}} &= 0 \quad \Rightarrow \quad \bf{w} = \sum_{n=1}^N a_n t_n \boldsymbol{\phi}(\bf{x}_n) \\
	\frac{\partial L}{\partial b} &= 0 \quad \Rightarrow \quad \sum_{n=1}^N a_n t_n = 0\\
	\frac{\partial L}{\partial \xi_n} &= 0  \quad \Rightarrow \quad a_n = C -\mu_n.
\end{align}
Using these results to eliminate $\bf{w}$, $b$ and $\{\xi_n\}$ from the Lagrangian, we obtain the dual Lagrangian in the form
\begin{equation}
	\tilde{L}(\bf{a}) = \sum_{n=1}^N a_n -\frac{1}{2} \sum_{n=1}^N \sum_{m=1}^N a_n a_m t_n t_m k(\bf{x}_n,\bf{x}_m).
\end{equation}
This is an identity completely comprised of Lagrangian multipliers, which is identical to the separable case, except that the constraints are different:
\begin{align}
	0 \leq a_n \leq C \\
	\sum_{n=1}^N a_n t_n = 0
\end{align}
The resulting marginal classifier can be written as
\begin{equation}
	y(\bf{x}) = \sum_{n=1}^N a_n t_n k(\bf{x,x}_n) + b
\end{equation} From which we assert that data points with $a_n = 0$ play no role in making predictions and hence the remaining data points are called **support vectors**.


#### 7.1.2 Compared with Logistic Regression
Here we briefly compare the SVM with the logistic regression model. The key difference is that the flat region in $E_{SV}(yt)$ leads to sparse solutions.

#### 7.1.3 Regression
To obtain sparse solutions, the quadratic error function is replaced by an insensitive error function <cite>Vapnik, 1995</cite>. The total error function is 
\begin{equation}
C \sum_{n=1}^{N}(\xi_n + \hat{\xi}_n) + \frac{1}{2} ||\bf{w}||^2.
\end{equation}
Adopt the Lagrangian
\begin{align}
	L = & C \sum_{n=1}^N (\xi_n + \hat{\xi}_n) + \frac{1}{2} ||\bf{w}||^2 - \sum_{n=1}^N (\mu_n\xi_n + \hat{\mu}_n\hat{\xi}_n) \\
	 &- \sum_{n=1}^N a_n(\epsilon + \xi_n + y_n -t_n) - \sum_{n=1}^N \hat{a}_n(\epsilon+\hat{\xi}_n-y_n+t_n).
\end{align}
Then we set the derivatives of the Lagrangian with respect to $\bf{w}$, $b$, $\xi_n$, and $\hat{\xi}_n$ to zero, giving
\begin{align}
	\frac{\partial L}{\partial \bf{w}} &= 0 \quad \Rightarrow \quad \bf{w} = \sum_{n=1}^N (a_n-\hat{a}_n)\boldsymbol{\phi}(\bf{x}_n)\\
	\frac{\partial L}{\partial b} &=0 \quad \Rightarrow \quad \sum_{n=1}^N(a_n-\hat{a}_n) = 0 \\
	\frac{\partial L}{\partial \xi_n} &= 0 \quad \Rightarrow \quad a_n+\mu_n = C \\
	\frac{\partial L}{\partial \hat{\xi}_n} &= 0 \quad \Rightarrow \quad \hat{a}_n+\hat{\mu}_n = C.
\end{align}
Using these results to simplify the Lagrangian, we obtain
\begin{align}
	\tilde{L}(\bf{a},\hat{\bf{a}}) = &-\frac{1}{2} \sum_{n=1}^N \sum_{m=1}^N (a_n-\hat{a}_n)(a_m-\hat{a}_m)k(\bf{x}_n,\bf{x}_m) \\
	&- \epsilon \sum_{n=1}^N (a_n+\hat{a}_n) + \sum_{n=1}^N (a_n - \hat{a}_n)t_n
\end{align}
and again we have the box constraints
\begin{align}
	0 \leq a_n &\leq C \\
	0 \leq \hat{a}_n &\leq C \\
	\sum_{n=1}^N (a_n - \hat{a}_n) &= 0.
\end{align}
The corresponding KKT conditions are given by
\begin{align}
	a_n(\epsilon + \xi_n + y_n -t_n) &= 0\\
	\hat{a}_n(\epsilon + \hat{\xi}_n-y_n+t_n) &= 0\\
	(C - a_n)\xi_n &= 0\\
	(C - \hat{a}_n) \hat{\xi}_n &= 0.
\end{align}
The predictions are given by
\begin{equation}
	y(\bf{x}) = \sum_{n=1}^N (a_n - \hat{a}_n)k(\bf{x},\bf{x}_n) +b.
\end{equation}
This is the key idea of SVM regression. All points within the tube have $a_n = \hat{a_n} = 0$. We again have a sparse solution, and the only terms that have to be evaluated in the predictive model are those that involve the support vectors.


### 7.2 RVM
The relevance vector machine or RVM <cite>(Tipping, 2001)</cite> is a Bayesian sparse kernel technique for regression and classification that shares many of the characteristics of the SVM whilst avoiding its principal limitations. Additionally, it typically leads to much sparser models resulting in correspondingly faster performance on test data whilst maintaining comparable generalization error.

#### 7.2.1 Regression
The model is:
\begin{align}
	y(\bf{x}) &= \sum_{n=1}^{N} w_n k(\bf{x},\bf{x}_n) + b\\
	p(\bf{w}|\boldsymbol{\alpha}) &= \prod_{i=1}^M \cal{N}(w_i |0,\alpha_i^{-1})
\end{align}
The key difference is that in RVM we have hyperparameters $\alpha_i$ for each of the weight parameter $w_i$, while in linear regression, all $w_i$ share one hyperparameter $\alpha$. 
Then the derivation is quite similar to the linear regression. We have:
\begin{equation}
p(w|t,X,\alpha, \beta) = \cal{N}(w|m.\Sigma)
\end{equation} where $m = \beta\Sigma\Phi^{\intercal}t$ and $\Sigma = (A + \beta\Phi^{\intercal}\Phi)^{-1}$
And we can adopt the evidence approximation to find the best hyperparameters $\\alpha$ and $\beta$, so that:
\begin{equation}
p(t|X, \alpha, \beta) = \int p(t|X, w, \beta)p(w|\alpha) dw
\end{equation}


As a result of the optimization, we find that a proportion of the hyperparameters $\{\alpha_i\}$ are driven to large (in principle infinite) values, and so the weight parameters wi corresponding to these hyperparameters have posterior distributions with mean and variance both zero. Thus those parameters, and the corresponding basis functions $\phi_i(x)$, are removed from the model and play no role in making predictions for new inputs. In the case of models of the form, the inputs $x_n$ corresponding to the remaining nonzero weights are called relevance vectors, because they are identified through the mechanism of automatic relevance determination, and are analogous to the support vectors of an SVM. 


RVM can be viewed as an example to use **automatic relevance determination** in the problem expressed as an adaptive linear combination of basis functions.

#### 7.2.2 Classification
We again introduce an ARD prior and the followings are similar to what we have discussed in logistic regression. The maximization of the evidence function, which is paramount for automatic relevance determination, is specified here.

We already have the parameters of the Laplace approximation to the posterior distribution of $\bf{w}$, given by
\begin{align}
	\bf{w}^* &= \bf{A}^{-1} \boldsymbol{\Phi}^{\intercal} (\bf{t-y}) \\
	\boldsymbol{\Sigma} &= (\boldsymbol{\Phi}^{\intercal} \bf{B} \boldsymbol{\Phi} + \bf{A})^{-1}
\end{align}
where $\bf{A} = {diag}(\alpha_i)$ and $\bf{B} = {diag}\{y_n(1-y_n)\}$. 

From (\ref{LCEvi}) we obtain
\begin{equation}
	p(\bf{t}|\alpha) \simeq p(\bf{t|w}^*)p(\bf{w}^*|\boldsymbol{\alpha})(2\pi)^{M/2}|\boldsymbol{\Sigma}|^{1/2}.
\end{equation}
Setting the derivatives of the marginal likelihood with respect to $\alpha_i$ to zero, we obtain
\begin{equation}
	-\frac{1}{2}(w_i^*)^2 + \frac{1}{2\alpha_i}-\frac{1}{2}\Sigma_{ii} = 0.
\end{equation}
Defining $\gamma_i = 1-\alpha_i \Sigma_{ii}$ and rearranging then gives
\begin{equation}
	\alpha_i^{new} = \frac{\gamma_i}{(w_i^*)^2}.
\end{equation}



If we define
\begin{equation}
	\hat{\bf{t}} = \boldsymbol{\Phi} \bf{w}^* + \bf{B}^{-1}(\bf{t-y})
\end{equation}
we can write the approximate log marginal likelihood in the form
\begin{equation}
	\ln p(\bf{t}|\boldsymbol{\alpha}) = -\frac{1}{2}\left\{ N \ln(2\pi) + \ln |\bf{C}| + (\hat{\bf{t}})^{\intercal} \bf{C}^{-1} \hat{\bf{t}} \right\}
\end{equation}
where
\begin{equation}
	\bf{C=B}+\boldsymbol{\Phi}\bf{A}\boldsymbol{\Phi}^{\intercal}.
\end{equation}
This takes the same form as the evidence function  in the regression case, and so we can apply the same analysis of sparsity and obtain the same fast learning algorithm in which we fully optimize a single hyperparameter $\alpha_i$ at each step.

#### 7.2.3  Conclusion
The principal disadvantage of the RVM is the relative **long training time** compared with the SVM. This is offset, however, by **the avoidance of cross-validation** runs to set the model complexity parameters. Furthermore, because it yields **sparser** models, the computation time on test points, which is usually the more important consideration in practice, is typically much less.

## 8. Graphical Models
A graph comprises nodes connected by links. In a probabilistic graphical model, each node represents a random variable (or group of random variables), and the links express probabilistic relationships between these variables. The graph then captures the way in which the joint distribution over all of the random variables can be decomposed into a product of factors each depending only on a subset of the variables.

### 8.1 Bayesian Networks
Bayesian network is also called directed graphical model, where the links have a particular directionality indicated by arrows. For each conditional distribution we add directed links (arrows) to the graph from the nodes corresponding to the variables on which the distribution is conditioned. The joint distribution defined by a graph is given by the product, over all of the nodes of the graph, of a conditional distribution for each node conditioned on the variables corresponding to the parents of that node in the graph.
\begin{equation}
	p(\bf{x}) = \prod_{k=1}^K p(x_k|pa_k)
\end{equation} where $p_{a_k}$ denotes the set of parents of $x_k$. This key equation expresses the factorization properties of the joint distribution for a directed graphical model. Note that no directed cycles are allowed in this model. 


With a DAG, we can take samples using ancestral sampling. The graphical model captures the causal process (Pearl, 1988) by which the observed data was generated. For this reason, such models are often called generative models. The hidden variables in a probabilistic model need not, however, have any explicit physical interpretation but may be introduced simply to allow a more complex joint distribution to be constructed from simpler components.

#### 8.1.1 Discrete variables
 The total number of parameters that must be specified for an arbitrary joint distribution over $M$ variables is $K_M − 1$ and therefore grows exponentially with the number $M$ of variables. From a graphical perspective, we can reduce the number of parameters by dropping links in the graph, at the expense of having a restricted class of distributions. An alternative way to reduce the number of independent parameters in a model is by sharing parameters. 
 
#### 8.1.2 Linear-Gaussian models
 We would like to show how a multivariate Gaussian can be expressed as a directed graph corresponding to a linear-Gaussian model over the component variables.
 
 Consider an arbitrary directed acyclic graph over $D$ variables in which node $i$ represents a single continuous random variable $x_i$ having a Gaussian distribution. And it's distribution is given by:
 \begin{equation}
 p(x_i|pa_i) = \cal{N}(x_i | \sum_{j\in pa_i} w_{ij}x_j + b_i, v_i)
 \end{equation}
 Therefore, the log of the joint distribution is given by:
 \begin{equation}
 ln p(\bf{x}) = \sum_{i=1}^D lnp(x_i|pa_i) = -\sum_{i=1}^D \frac{1}{2v_i}(x_i - \sum_{j\in pa_i}w_{ij}x_j - b_i)^2 + const
 \end{equation}
 We can see that this is a quadratic function of the components of $\bf{x}$, and hence the joint distribution $p(x)$ is a multivariate Gaussian.
 
 
 We can determine the mean and covariance of the joint distribution recursively as follows. Each node's mean is depended on the set $pa_i$, which gives:
 \begin{equation}
 E[x_i] = \sum_{j\in pa_i}w_{ij}E[x_j] + b_i
 \end{equation}
 Thus we can find the components of $E[\bf{x}] = (E[x_1 ], . . . , E[x_D ])^{\intercal} $by starting at the lowest numbered node and working recursively through the graph. Similarly, we can use the $i, j$ element of the covariance matrix for $p(x)$ in the form of a recursion relation:
 \begin{equation}
 cov[x_i, x_j] = \sum_{k\in pa_j} w_{jk}cov[x_i, x_k] + I_{ij}\sqrt{v_i}\sqrt{v_j}
 \end{equation}
 
 
 We can readily extend the linear-Gaussian graphical model to the case in which the nodes of the graph represent multivariate Gaussian variables.
 \begin{equation}
 p(\bf{x_i}|pa_i) = \cal{N}(x_i | \sum_{j \in pa_i}W_{ij}x_j + b_i, \Sigma_i)
 \end{equation} where $x_i$ and $x_j$ are random vectors. 
 
 
### 8.2 Markov random fields
A Markov random field is described by an undirected graph.

The Markov random field and its corresponding undirected graph is created for the purpose of simplifying the derivations of conditional independence. Because there is no direction specified on the links, the concept of ``blocked'' is straightforward and without annoying exceptions as in the case of Bayesian networks. Thus, if we consider two nodes $x_i$ and $x_j$ that are not connected by a link, then these variables must be conditionally independent given all other nodes in the graph. This conditional independence property can be expressed as
\begin{equation}
	p(x_i,x_j|\bf{x}_{\backslash \{i,j\}}) = p(x_i|\bf{x}_{\backslash\{i,j\}}) p(x_j|\bf{x}_{\backslash
	\{i,j\}}).
\end{equation}
Note that $p(x_i|\bf{x}_{\backslash\{i,j\}})$ is independent of $x_j$ and $p(x_j|\bf{x}_{\backslash
	\{i,j\}})$ is independent of $x_i$. As a result, the factorization of the joint distribution must therefore be such that $x_i$ and $x_j$ do not appear in the same factor in order for the conditional independence property to hold for all possible distributions belonging to the graph.

The joint distribution defined by the Markov random fields can thus be written as a product of potential functions $\psi_C(\bf{x}_C)$ over the maximal cliques of the graph
\begin{equation}
	p(\bf{x}) = \frac{1}{Z} \prod_C \psi_C(\bf{x}_C)
\end{equation} where we denote a maximal clique by $C$. Here the quantity $Z$, sometimes called the partition function, is a normalization constant and is given by
\begin{equation}
	Z = \sum_{\bf{x}} \prod_C \psi_C(\bf{x}_C)
\end{equation}

Since we are restricted to potential functions which are strictly positive it is convenient to express them as exponentials, so that
\begin{equation}
	\psi_C(\bf{x}_C) = \exp \{ -E(\bf{x}_C) \}
\end{equation}
where $E(\bf{x}_C)$ is called an energy function, and thee exponential representation is called the Boltzmann distribution.

The factors in the joint distribution for a directed graph have a specific probabilistic interpretation. For an undirected graph, the potential functions can be viewed as expressing which configurations of the local variables are preferred to others, i.e., the potential functions should be relatively smaller when the probabilities of the configurations they represent are bigger.

 
#### 8.2.1 Relation to directed graphs
Let us consider how to concert any distribution specified by a factorization over a directed graph into one specified by a factorization over an undirected graph. In order to achieve this, we must ensure that the set of variables that appears in each of the conditional distributions is a member of at least one clique of the undirected graph. For nodes on the directed graph having just one parent, this is achieved simply by replacing the directed link with an undirected link. For nodes having more than one parent, however, we should add extra links between all pairs of parents of these nodes. Anachronistically, this process of ``marrying the parents'' has become known as moralization, and the resulting undirected graph, after dropping the arrows, is called the moral graph. The conditional independence properties may loss in this procedure. However, the process of moralization adds the fewest extra links and so retains the maximum number of independence properties.
 
 
### 8.3 Factor graphs
Both directed and undirected graphs allow a global function of several variables to be expressed as a product of factors over subsets of those variables. **Factor graphs** make this decomposition explicit by introducing additional nodes for the factors themselves in addition to the nodes representing the variables.


We write the joint distribution over a set of variables in the form of a product
of factors
\begin{equation}
	p(\bf{x}) = \prod_{s} f_s(\bf{x}_s)
\end{equation}
where $\bf{x}_s$ denotes a subset of the variables. In a factor graph, there is a node (depicted as usual by a circle) for every variable in the distribution, as was the case for directed and undirected graphs. There are also additional nodes (depicted by small squares) for each factor fs(xs) in the joint distribution. Finally, there are undirected links connecting each factor node to all of the variables nodes on which that factor depends. For example, the factor graph of the distribution $p(x) = f_a(x_1, x_2)f_b(x_1,x_2)f_c(x_3)$ is
![屏幕快照 2019-03-11 上午8.15.14.png](resources/72E06658DF46FC52B309684B1A414645.png =293x167)
Factor graphs are said to be bipartite because they consist of two distinct kinds of nodes, and all links go between nodes of opposite type.

### 8.4 Conditional independence
We first introduced three kind of graph model, then we move on to the most important property in graphical models --- conditional independence. If we find ways to inference the conditional independence property in a graphical model, the number of the parameters can be sharply reduced. 

#### 8.4.1 Directed graphs

##### 8.4.1.1 Three examples
![屏幕快照 2019-03-11 上午8.27.51.png](resources/B684EE3A439078A12D19C0B1185DE8BB.png =215x146)
The first structure is called tail-to-tail. The joint distribution corresponding to this graph can be written as following:
\begin{equation}
p(a,b,c) = p(a|c)p(b|c)p(c)
\end{equation}
If none of the variables is observed, the conditional independence is not held for node $a$ and node $b$. While the node $c$ is observed, the conditional distribution of $a,b$, given $c$ can be written as:
\begin{equation}
p(a,b|c) = p(a|c)p(b|c)
\end{equation} which shows that given c, node a and node b are conditionally independent. When we condition on node c, the conditioned node ‘blocks’ the path from a to b and causes a and b to become (conditionally) independent.
 

![屏幕快照 2019-03-11 上午8.30.29.png](resources/F4B04FFC420863612C7DF20F3DF2FE74.png =214x103)
 Shown in figure 2, the second one is called head-to-tail, and the joint distribution related to it is:
 \begin{equation}
 p(a,b,c) = p(a)p(c|a)p(b|c)
 \end{equation}
 We cannot obtain conditional independence property when nothing is observed. However, when observing node c, node a is conditionally independent to node b. This observation blocks the path from a to b.
 
 
 ![屏幕快照 2019-03-11 上午8.33.42.png](resources/D40A5B06602CFCB8DCD352141F978335.png =212x148)
 The last one is head-to-head shown in figure 3. The joint distribution is:
 \begin{equation}
 p(a,b,c) = p(a)p(b)p(c|a,b)
 \end{equation}
 When nothing has been observed, node a and node b is conditionally independent. But when node c is observed, they are not independent. The observation of node c unblocks the path from node a to node b.
 
##### 8.4.1.2 D-seperation
The general scheme for ascertaining conditional independence is called d-separation. Consider a general directed graph in which A, B, and C are arbi- trary nonintersecting sets of nodes (whose union may be smaller than the complete set of nodes in the graph).  Suppose we want to know whether a particular conditional independence statement $A \pm B|C$ is implied by a given directed acyclic graph. To do so, we consider all possible paths from any node in $A$ to any node in $B$. Any such path is said to be **blocked** if it includes a node such that either 
* the arrows on the path meet either head-to-tail or tail-to-tail at the node, and the node is in the set $C$, or
* the arrows meet head-to-head at the node, and neither the node, nor any of its descendants, is in the set $C$.

If all paths are blocked, then $A$ is said to be d-separated from $B$ by $C$, and the joint distribution over all of the variables in the graph will satisfy $A\pm B|C$.

#### 8.4.2 Undirected graph
For undirected graph, suppose we want to identify three sets of nodes, denoted $A,B$ and $C$, and that we consider the conditional independence property
\begin{equation}
	A \pm B|C.
\end{equation}
To test whether this property is satisfied by a probability distribution defined by a graph we consider all possible paths that connect nodes in set $A$ to nodes in set $B$. If all such paths pass through one or more nodes in $C$, then all such paths are blocked and so the conditional independence property holds. This is exactly the same as the d-separation criterion except that there is no ``explaining'' away phenomenon.

### 8.5 Inference in Graphical Models

#### 8.5.1 Sum-product algorithm
In a factor graph model, the joint distribution can be described as
\begin{equation}
	p(\bf{x}) = \prod_{s \in ne(x)} F_s(x,X_s)
\end{equation}
The marginal distribution is obtained by
\begin{align}
	p(x) &= \sum_{\bf{x}\backslash x} p(\bf{x}) \notag \\
	&= \prod_{s \in ne(x)} \left[ \sum_{X_s} F_s(x,X_s) \right] \notag \\
	&= \prod_{s \in ne(x)} \mu_{f_s \rightarrow x}(x).
\end{align}

From 
\begin{equation}
	\mu_{f_s \rightarrow x}(x) \equiv \sum_{X_s} F_s(x,X_s)
\end{equation}
and 
\begin{equation}
	F_s(x,X_s) = f_s(x,x_1,\cdots,x_M) G_1(x_1,X_{s1}) \cdots G_M(x_M,X_{sM})
\end{equation}
we obtain
\begin{align}
	\mu_{f_s \rightarrow x}(x) &= \sum_{x_1}\cdots \sum_{x_M} f_s(x,x_1,\cdots,x_M)\prod_{m \in ne(f_s) \backslash x}\left[\sum_{X_{sm}}G_m(x_m,X_{sm}) \right]\notag \\
	&= \sum_{x_1}\cdots \sum_{x_M} f_s(x,x_1,\cdots,x_M) \prod_{m \in ne(f_s) \backslash x}\mu_{x_m \rightarrow f_s}(x_m).
\end{align}
where we define
\begin{equation}
	\mu_{x_m \rightarrow f_s}(x_m)\equiv \sum_{X_{sm}} G_m(x_m, X_{sm}).
\end{equation}

From the tree structure property, we obtain
\begin{equation}
	G_m(x_m, X_{sm}) = \prod_{l \in ne(x_m)\backslash f_s} F_l(x_m,X_{ml}).
\end{equation}
As a result,
\begin{align}
	\mu_{x_m \rightarrow f_s} (x_m) &= \prod_{l \in ne(x_m)\backslash f_s}\left[ \sum_{X_{ml}} F_l(x_m,X_{ml}) \right]\notag \\
	&= \prod_{l \in ne(x_m) \backslash f_s} \mu_{f_l \rightarrow x_m}(x_m).
\end{align}

To start the recursion, we need to view the messages sent by leaf nodes. If a leaf node is a variable node, then the message that it sends along its one and only link is given by
\begin{equation}
	\mu_{x \rightarrow f}(x) = 1.
\end{equation} 
Similarly, if the leaf node is a factor node, we see that the message sent should take the form
\begin{equation}
	\mu_{f \rightarrow x}(x) = f(x).
\end{equation}

Now suppose we wish to find the marginals for \imp{every variable node} in the graph. There is an efficient method for this. Arbitrarily pick any (variable or factor) node and designate it as the root. Propagate messages from the leaves to the root as before. These in turn will then have received messages from all of their neighbors and so can send out messages along the links going away from the root, and so on. By now, a message will have passed in both directions across every link in the graph, and every node will have received a message from all of its neighbors and we can readily calculate the marginal distribution for every variable in the graph.

For graphs without a tree structure, an approximate procedure named Loopy belief propagation may be applied.

#### 8.5.2 Max-sum algorithm
The max-sum algorithm is an efficient method to find a setting of the variables that has the largest probability and to find the value of that probability on graphical models of tree structures. The structure of max-sum algorithm is identical to that of the sum-product algorithm.

The max operator satisfies the distributive law, i.e.
\begin{equation}
	\max(ab,ac) = a \max(b,c)
\end{equation}
for $a\geq 0$(as will always be the case for the factors in a graphical model). This allows us to exchange products with maximizations analogously.

In practice, products of many small probabilities can lead to numerical underflow problems, and so it is convenient to work with the logarithm of the joint distribution. Note that
\begin{align}
	\ln(\max_{\bf{x}}p(\bf{x})) &= \max_{\bf{x}} \ln p(\bf{x})\\
	\max(a+b,a+c) &= a + \max(b,c),
\end{align}
so the distributive property is preserved.
Analogous to sum-product algorithm, we have the formulae of messages:
\begin{align}
	\mu_{f\rightarrow x}(x) &= \max_{x_1,\cdots,x_M}\left[\ln f(x,x1,\cdots,x_M)+\sum_{m \in ne(f)\backslash x} \mu_{x_m \rightarrow f}(x_m)\right]\\
	\mu_{x\rightarrow f}(x) &= \sum_{l \in ne(x)\backslash f}\mu_{f_l \rightarrow x}(x).
\end{align}
The initial messages sent by the leaf nodes are obtained by analogy with sum-product algorithm and are given by
\begin{align}
	\mu_{x \rightarrow f}(x) &= 0\\
	\mu_{f \rightarrow x}(x) &= \ln f(x)
\end{align}
while at the root node the maximum probability can then be computed, using
\begin{equation}
	p^{max} = \max_{x} \left[\sum_{s \in ne(x)}\mu_{f_s \rightarrow x}(x)\right].
\end{equation}


In order to determine the parameters which maximize the joint probability, we need to employ a special skill called back-tracking. If a message is sent from a factor node $f$ to a variable node $x$, a maximization is performed over all other variable nodes $x_1,\cdots,x_M$ that are neighbors of that factor node. When we perform this maximization, we keep a record of which values of the variables $x_1,\cdots,x_M$ gave rise to the maximum. Then in the back-tracking step, having found $x^{max}$, we can then use these stored values to assign consistent maximizing states $x_1^{max},\cdots,x_M^{max}$. The max-sum algorithm, with back-tracking, gives an exact maximizing configuration for the variables provided the factor graph is a tree.

There are some variants of this max-sum algorithm. In the particular context of the hidden Markov model, this is known as the forward-backward algorithm(or the Baum-Welch algorithm) and Viterbi algorithm.

For general graphical models without a tree structure, approximations are needed. A simple idea, which is called Loopy belief propagation, is simply to apply the algorithm even though there is no guarantee that it will yield good results. Each message sent from a node replaces any previous message sent in the same direction across the same link and will itself be a function only of the most recent messages received by that node at previous steps of the algorithm.

ps: this part referred to <cite>Yang Song's notes</cite>

## 9. Mixture Models and EM
If we define a joint distribution over observed and latent variables, the correspond- ing distribution of the observed variables alone is obtained by marginalization. This allows relatively complex marginal distributions over observed variables to be expressed in terms of more tractable joint distributions over the expanded space of observed and latent variables.

### 9.1 K-means Clustering
K-means clustering is a nonprobabilistic technique to find clusters in a set of data points. Consider the problem of identifying groups, or clusters, of data points in a multidimensional space. Suppose we have a data set $\{x_1 , . . . , x_N \}$ consisting of N observations of a random D-dimensional Euclidean variable $x$. Our goal is to partition the data set into some number $K$ of clusters and find the mean $\mu_i$ of each cluster $i$, where we shall suppose for the moment that the value of $K$ is given.


The distortion measure takes the form:
\begin{equation}
J = \sum_{n=1}^K\sum_{k=1}^Kr_{nk}|x_n - \mu_k|^2
\end{equation}
We would like to find the value of $\{r_{nk}\}$ and $\{\mu_k\}$ to minimize the distortion measure. First we choose some initial values for the $\mu_k$ . Then in the first phase we minimize $J$ with respect to the $r_{nk}$, keeping the μk fixed. In the second phase we minimize $J$ with respect to the $\mu_k$, keeping $r_{nk}$ fixed. This two-stage optimization is then repeated until convergence.
![屏幕快照 2019-03-11 下午1.48.15.png](resources/769024008BE2D894350C6D87AFA70E0F.png =303x69)
![屏幕快照 2019-03-11 下午1.48.28.png](resources/9525F45FE6C93E2CB339563B264D7E17.png =164x60)
We will show that these  two stages correspond to the E and M step in the EM algorithm. 


Note that K-means clustering is easy to be influenced by outliers.  It is also worth noting that the K-means algorithm itself is often used to initialize the parameters in a Gaussian mixture model before applying the EM algorithm.  


### 9.2 Gaussian mixture model (GMM)
The Gaussian mixture model can be written as a linear superposition of Gaussians as :
\begin{equation}
	p(\bf{x}) = \sum_{k=1}^K \pi_k \cal{N}(\bf{x}|\boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k).
\end{equation}
We now introduce a $K$-dimensional binary random **latent** variable $\bf{z}$ having a 1-of-$K$ representation in which a particular element $z_k$ is equal to 1 and all other elements are equal to 0. As a result
\begin{align}
 p(\bf{z}) &= \prod_{k=1}^{K} \pi_k^{z_k} \\
 p(\bf{x}|z_k=1) &= \cal{N}(\bf{x}|\boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k) \\
 p(\bf{x|z})&= \prod_{k=1}^K \cal{N}(\bf{x}|\boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k)^{z_k}\\
 p(\bf{x}) &= \sum_{\bf{z}} p(\bf{z})p(\bf{x|z}) = \sum_{k=1}^K \pi_k \cal{N}(\bf{x}|\boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k).
\end{align}
Note that there are only $k$ $z$s, since it is of one-hot encoding. With help of the latent variable, we make the Gaussian mixture into a marginal distribution of $p(x,z)$. 


In addition, we use $\gamma(z_k)$ to denote $p(z_k=1|\bf{x})$, whose value can be found using Bayes' theorem
\begin{align}
	\gamma(z_k)\equiv p(z_k=1|\bf{x}) &= \frac{p(z_k=1)p(\bf{x}|z_k=1)}{\sum_{j=1}^K p(z_j=1)p(\bf{x}|z_j=1)} \\
	&= \frac{\pi_k \cal{N}(\bf{x}|\boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k)}{\sum_{j=1}^K \pi_j \cal{N}(\bf{x}|\boldsymbol{\mu}_j,\boldsymbol{\Sigma}_j)}.
\end{align}
$\gamma(z_k)$ can also be viewed as the **responsibility** that component $k$ takes for 'explaining' the observation $\bf{x}$.
![屏幕快照 2019-03-11 下午2.22.27.png](resources/38B2866016051B81B25D8672174A81F5.png =577x179)

#### 9.2.1 Maximum likelihood
The log of the likelihood function is given by
\begin{equation}
	\ln p(\bf{X}|\boldsymbol{\pi,\mu,\Sigma}) = \sum_{n=1}^N \ln \left\{ \sum_{k=1}^K \pi_k \cal{N}(\bf{x}_n|\boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k) \right\}.
\end{equation}
There are two points to take care of. Firstly, the likelihood may have severe singularities, which may happen when a single Gaussian collapses onto a data point. However, this will not happen in the single Gaussian distribution, since when the collapse happens, the possibility of other points will go zero.  Suppose a Gaussian has the covariance matrix $\boldsymbol{\Sigma}_k = \sigma_k^2 \bf{I}$, and its mean $\boldsymbol{\mu}_j$ exactly equals to one of the data points so that $\boldsymbol{\mu}_j = \bf{x}_n$. The data point will then contribute a term in the likelihood of the form
\begin{equation}
	\cal{N}(\bf{x}_n|\bf{x}_n,\sigma_j^2 \bf{I}) = \frac{1}{(2\pi)^{1/2}}\frac{1}{\sigma_j}
\end{equation}
which will go to infinity in the limit $\sigma_j \rightarrow 0$. As a result, a likelihood having such singularities can never be optimized since it can be as large as it wishes to be. This is a unique phenomenon for mixtures. We shall see that this difficulty does not occur if we adopt a Bayesian approach.


Secondly, for any given given maximum likelihood solution, a K-component mixture will have a total of $K!$ equivalent solutions corresponding to the $K!$ ways of assigning $K$ sets of parameters to $K$ components. This problem is called **identifiability** and should be taken into account in the case of model comparison.


Maximizing the log likelihood function for a Gaussian mixture model turns out to be a more complex problem than for the case of a single Gaussian. The difficulty arises from the presence of the summation over k that appears inside the logarithm in, so that the logarithm function no longer acts directly on the Gaussian. In this chapter, we will adopt EM algorithm to solve the problem. 

#### 9.2.2 EM algorithm for GMM
![屏幕快照 2019-03-11 下午3.13.19.png](resources/18D2090E5494C4BF14920DDF97EB93E1.png =580x275)
![屏幕快照 2019-03-11 下午3.13.38.png](resources/D041944489130DE886DB60EC4D130B4A.png =570x437)
By setting the derivative over $\mu_k$ and $\pi_k$ to be zero, we have:
\begin{equation}
\mu_k = \frac{1}{N_k}\sum_{n=1}^N\gamma(z_{nk})x_n
\end{equation} and
\begin{equation}
\pi_k = \frac{N_k}{N}
\end{equation}
However, the $\gamma(z_{nk})$ depends on the $\mu_k$ and $\pi_k$, which makes it impossible to find the analytic solution. We can use EM algorithm to solve the problem. 

### 9.3 EM algorithm
We present a complementary view of the EM algorithm that recognizes the key role played by latent variables.


The log likelihood is given by:
\begin{equation}
ln p(X|\theta) = ln \{\sum_Zp(X, Z|\theta)\}
\end{equation}
We shall call ${X, Z}$ the complete data set, and we shall refer to the actual observed data $X$ as incomplete. Because we cannot use the complete-data log likelihood, we consider instead its expected value under the posterior distribution of the latent variable, which corresponds (as we shall see) to the E step of the EM algorithm. In the subsequent M step, we maximize this expectation.
![屏幕快照 2019-03-11 下午3.36.28.png](resources/045E749686FCCD0C4FB5D31D583BC0CB.png =571x218)
The general EM algorithm is:
![屏幕快照 2019-03-11 下午3.37.43.png](resources/43CAE441E2D7007F0810C5B2FBBE1373.png =558x136)
![屏幕快照 2019-03-11 下午3.38.14.png](resources/87B93E148CCECD650866830EACEAB70D.png =564x308)
Briefly, we use the posterior distribution over the latent variable to calculate the expectation over the parameter $\theta$. 

#### 9.3.1 EM for GMM again
The log joint distribution of the complete data is 
\begin{equation}
	\ln p(\bf{X,Z}|\boldsymbol{\mu,\Sigma,\pi}) = \sum_{n=1}^N \sum_{k=1}^K z_{nk}\{ \ln \pi_k +\ln \cal{N}(\bf{x}_n|\boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k)\}
\end{equation}

Maximize this joint distribution with respect to $\pi_k$ directly by Lagrangian multipliers, we obtain
\begin{equation}
	\pi_k = \frac{1}{N} \sum_{n=1}^N z_{nk}.
\end{equation}

**The E step**. The expected value of the complete-data log likelihood function is therefore given by
$$
\begin{equation}
	\mathbb{E}_{\bf{Z}}[\ln p(\bf{X,Z}|\boldsymbol{\mu,\Sigma,\pi})] = \sum_{n=1}^N \sum_{k=1}^K \mathbb{E}[z_{nk}]\{ \ln \pi_k + \ln \cal{N}(\bf{x}_n|\boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k) \} 
\end{equation}
$$
where the expected value of the posterior variable $z_{nk}$ under this posterior distribution is then given by
\begin{align}
	\mathbb{E}[z_{nk}] &= \frac{\sum_{\bf{z}_n} z_{nk}\prod_{k'}[ \pi_{k'} \cal{N}(\bf{x}_n|\boldsymbol{\mu}_{k'},\boldsymbol{\Sigma}_{k'})]^{z_{nk'}}}{\sum_{\bf{z}_n}\prod_j[\pi_j \cal{N}(\bf{x}_n|\boldsymbol{\mu}_j,\boldsymbol{\Sigma}_j)]^{z_{nj}}} \\
	&= \frac{\pi_k \cal{N}(\bf{x}_n|\boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k)}{\sum_{j=1}^K \pi_j \cal{N}(\bf{x}_n|\boldsymbol{\mu}_j,\boldsymbol{\Sigma}_j)}=\gamma(z_{nk})
\end{align}
**The M step**. Maximize directly with respect to $\boldsymbol{\mu}_k$, $\boldsymbol{\Sigma}_k$ and $\pi_k$ leads to closed form solutions given by
\begin{align}
	\boldsymbol{\mu}_k^{new} &= \frac{1}{N_k} \sum_{n=1}^N \gamma(z_{nk})\bf{x}_n \\
	\boldsymbol{\Sigma}_k^{new} &= \frac{1}{N_k} \sum_{n=1}^N \gamma(z_{nk}) (\bf{x}_n - \boldsymbol{\mu}_k)(\bf{x}_n-\boldsymbol{\mu}_k)^{\intercal} \\
	\pi_k^{new} &= \frac{N_k}{N}.
\end{align}

#### 9.3.2 EM with K-means
Compared with K-mens, we can see that the K-means algorithm performs a hard assignment of data points to clusters, in which each data point is associated uniquely with one cluster, while the EM algorithm makes a soft assignment based on the posterior probabilities. In fact, we can derive the K-means algorithm as a particular limit of EM for Gaussian mixtures as follows.


Consider a Gaussian mixture model in which the covariance matrices of the mixture components are given by $\epsilon I$, where $\epsilon$  is a variance parameter that is shared by all of the components. 
\begin{equation}
p(x|\mu_k, \Sigma_k) = \frac{1}{(2\pi\epsilon)^{0.5}}exp\{-\frac{1}{2\epsilon}|x-\mu_k|^2\}
\end{equation}
And the responsibility takes the form:
\begin{equation}
\gamma(z_{nk}) = \frac{\pi_k\exp(-\frac{|x_n-\mu_k|^2}{2\epsilon})}{\sum_j\pi_j\exp(-\frac{|x_n - \mu_j|^2}{2\epsilon})}
\end{equation}
If we consider the limit  $\epsilon$ → 0, the responsibilities $\gamma(z_nk)$ for the data point $x_n$ all go to zero except for term $j$, for which the responsibility $\gamma(z_{nj})$ will go to unity, which is the same as the K-means algorithm. Under this setting, the result given by EM is just the result given by K-means. What's more, the K-means algorithm does not estimate the covariances of the clusters but only the cluster means. 




#### 9.3.3 Mixtures of Bernoulli distributions
We now discuss mixtures of discrete binary variables described by Bernoulli distributions. This model is also known as latent class analysis <cite>(Lazarsfeld and Henry, 1968; McLachlan and Peel, 2000)</cite>. 


Consider a set of $D$ binary variables $x_i$, which satisfy the Bernoulli distribution, we have:
\begin{equation}
    p(\bf{x}|\boldsymbol{\mu}) = \prod_{i=1}^D \mu_i^{x_i}(1-\mu_i)^{(1-x_i)}
\end{equation}
Now let us consider a finite mixture of these distributions given by
\begin{equation}
	p(\bf{x}|\boldsymbol{\mu,\pi}) = \sum_{k=1}^K \pi_k p(\bf{x}|\boldsymbol{\mu}_k).
\end{equation}
The log likelihood function for this model is given by
\begin{equation}
	\ln p(\bf{X}|\boldsymbol{\mu,\pi}) = \sum_{n=1}^N \ln \left\{ \sum_{k=1}^K \pi_k p(\bf{x}_n|\boldsymbol{\mu}_k) \right\}.
\end{equation}


The we focus on the EM algorithm to find the solution. We introduce the K-dimensional binary latent variable $bf{z}$:
\begin{equation}
    p(\bf{x|z},\boldsymbol{\mu}) = \prod_{k=1}^K p(\bf{x}|\boldsymbol{\mu}_k)^{z_k}
\end{equation} while the prior distribution for the latent variables is given by
\begin{equation}
	p(\bf{z}|\boldsymbol{\pi}) = \prod_{k=1}^K \pi_k^{z_k}.
\end{equation}

Now we move on to the **EM** algorithm. The complete-data log likelihood function is given by:
\begin{align}
	\ln p(\bf{X,Z}|\boldsymbol{\mu,\pi}) = \sum_{n=1}^{N} \sum_{k=1}^K z_{nk}  \left\{   \ln \pi_k 
	+ \sum_{i=1}^D [ x_{ni}\ln \mu_{ki}+(1-x_{ni})\ln(1-\mu_{ki}) ]  \right\} 
\end{align}

**The E step.** We take the expectation of the complete-data log likelihood with respect to the posterior distribution of the latent variables to give
\begin{equation}
	\mathbb{E}_{\bf{Z}}[\ln p(\bf{X,Z}|\boldsymbol{\mu,\pi})]=\sum_{n=1}^{N}\sum_{k=1}^K \gamma(z_{nk})\left\{\ln \pi_k + \sum_{i=1}^D [x_{ni}\ln \mu_{ki}+(1-x_{ni})\ln(1-\mu_{ki})] \right\}
\end{equation}
where
\begin{align}
	\gamma(z_{nk}) =  \mathbb{E}[z_{nk}] = \frac{\pi_k p(\bf{x}_n|\boldsymbol{\mu}_k)}{\sum_{j=1}^K \pi_j p(\bf{x}_n|\boldsymbol{\mu}_j)}.
\end{align}

**The M step** Setting the derivative of the expectation with respect to $\boldsymbol{\mu}_k$ equal to zero and rearrange the terms, we obtain
\begin{equation}
	\boldsymbol{\mu}_k = \bar{\bf{x}}_k = \frac{1}{N_k}\sum_{n=1}^N \gamma(z_{nk}) \bf{x}_n.
\end{equation}
Then we maximize with respect to $\pi_k$ and obtain
\begin{equation}
\pi_k = \frac{N_k}{N}= \frac{\sum_{n=1}^N \gamma(z_{nk})}{N}.
\end{equation}

#### 9.3.4 EM for Bayesian linear regression
Remain to be written.


#### 9.3.5 EM for RVM
Remain to be written. 

## 10. Approximate Inference
Remain to be written. 

## 11. Sampling Methods
We consider approximate inference methods based on numerical sampling, also known as Monte Carlo techniques.

### 11.1 Basic Sampling Algorithms
In this section, we consider some simple strategies for generating random samples from a given distribution.

#### 11.1.1 Standard distributions
We first consider how to generate random numbers from simple nonuniform distributions, assuming that we already have available a source of uniformly distributed random numbers. Suppose that $z$ is uniformly distributed over the interval (0, 1), and that we transform the values of $z$ using some function $f(·)$ so that $y = f(z)$. The distribution of $y$ will be governed by
\begin{equation}
p(y) = p(z)|\frac{dz}{dy}|
\end{equation} where $p(z) = 1$.
And then we have:
\begin{equation}
z = h(y) = \int_{\infty}^y p(\hat{y}) d\hat{y}
\end{equation}


Consider the exponential distribution, 
\begin{equation}
p(y) = \lambda \exp(-\lambda y)
\end{equation}
If we take the transformation $y = \lambda^{-1}ln(1-z)$, then variable $y$ is of exponential distribution. 


Consider the Cauchy distribution which takes the form:
\begin{equation}
p(y) = \frac{1}{\pi}\frac{1}{1+y^2}
\end{equation}
In this case, the inverse of the indefinite integral can be expressed in terms of the ‘tan’ function.


The generalization to multiple variables is straightforward and involves the Jacobian of the change of variables, so that
\begin{equation}
p(y_1, \dots, y_M) = p(z_1, \dots, z_M)|\frac{\partial(z_1, \dots, z_M)}{\partial(y_1, \dots, y_M)}|
\end{equation}


Obviously, the transformation technique depends for its success on the ability to calculate and then invert the indefinite integral of the required distribution. Such operations will only be feasible for a limited number of simple distributions. 

#### 11.1.2 Rejection sampling
The rejection sampling framework allows us to sample from relatively complex distributions, subject to certain constraints. We need a simpler distribution $q(z)$, sometimes called a **proposal distribution**, from which we can readily draw examples. We next introduce a constant $k$ which is the smallest value satisfying $kq(z) \geq \tilde{p}(z)$. The function $kq(z)$ is called the comparison function. Each step of the rejection sampler involves generating two random numbers. First, we generate a number $z_0$ from the distribution $q(z)$. Next, we generate a number $u_0$ from the uniform distribution under the curve of the function $kq(z)$. Finally, if $u_0 > \tilde{p}(z_0)$ then the sample is rejected, otherwise $u_0$ is retained.
Therefore, the probability that a sample will be accepted is given by:
\begin{equation}
p(accept) = \int \frac{\hat{p(z)}}{kq(z)}q(z)dz = \frac{1}{k}\int \hat{p(z)}dz
\end{equation}


* For many practical examples, where the desired distribution may be multimodal and sharply peaked, it will be extremely difficult to find a good proposal distribution and comparison function.
* The exponential decrease of acceptance rate with dimensionality is a generic feature of rejection sampling. As a result, rejection can only be a useful technique in one or two dimensions.

#### 11.1.3 Adaptive rejection sampling
In many cases, we can not find a suitable proposal distribution $q(z)$. Instead, we can construct the envelope function on the fly based on measured values of the distribution $p(z)$ <cite>(Gilks and Wild, 1992)</cite>. When $p(z)$ is log concave, construction of an envelope function will be particularly straightforward.


The function $\ln p(z)$ and its gradient are evaluated at some initial set of grid points, and the intersections of the resulting tangent lines are used to construct the envelop function. Next a sample value is drawn from the envelope distribution. The envelope distribution is a succession of linear functions, and hence the envelope distribution itself comprises a piecewise exponential distribution of the form
\begin{equation}
	q(z) = k_i \lambda_i \exp \{ -\lambda_i (z - z_i) \} \qquad \hat{z}_{i-1,i} < z \leq \hat{z}_{i,i+1}
\end{equation}
where $\hat{z}_{i-1,i}$ is the point of intersection of the tangent lines at $z_{i-1}$ and $z_i$, $\lambda_i$ is the slope of the tangent at $z_i$ and $k_i$ accounts for the corresponding offset.


Once a sample has been drawn, the usual rejection criterion can be applied. If the sample is accepted, then it will be a draw from the desired distribution. If, however, the sample is rejected, then it is incorporated into the set of grid points, a new tangent line is computed, and the envelope function is thereby refined.


Although rejection can be a useful technique in one or two dimensions it is unsuited to problems of high dimensionality. It can, however, play a role as a subroutine in more sophisticated algorithms for sampling in high dimensional spaces.


#### 11.1.4 Importance sampling
The technique of importance sampling provides a framework for approximating expectations directly but does not itself provide a mechanism for drawing samples from distribution $p(\bf{z})$. As in the case of rejection sampling, importance sampling is also based on the use of a proposal distribution $q(\bf{z})$ from which it is easy to draw samples. So the expectation can be expressed in the form
\begin{align}
	\mathbb{E}[f] &= \int f(\bf{z})p(\bf{z}) d \bf{z} \notag \\
	&= \frac{Z_q}{Z_p} \int f(\bf{z})\frac{\tilde{p}(\bf{z})}{\tilde{q}(\bf{z})}q(\bf{z}) d \bf{z} \notag \\
	&\simeq \frac{Z_q}{Z_p}\frac{1}{L} \sum_{l=1}^{L} \frac{\tilde{p}(\bf{z}^{(l)})}{\tilde{q}(\bf{z}^{(l)})} f(\bf{z}^{l}) \\
	&= \frac{Z_q}{Z_p}\frac{1}{L} \sum_{l=1}^{L} \tilde{r}_l f(\bf{z}^{l}).
\end{align} 
The quantities $r_l = p(\bf{z}^{(l)})/q(\bf{z}^{(l)})$ are known as **importance weights**, and they correct the bias introduced by sampling from the wrong distribution. We can use the same sample set to evaluate the ratio $Z_p/Z_q$ with the result
\begin{align}
	\frac{Z_p}{Z_q} &= \frac{1}{Z_q}\int \tilde{p}(\bf{z}) d \bf{z} = \int \frac{\tilde{p}(\bf{z})}{\tilde{q}(\bf{x})} q(\bf{z}) d \bf{z} \notag \\
	&\simeq \frac{1}{L} \sum_{l=1}^{L} \tilde{r}_l
\end{align}
where $\tilde{r}_l =  \tilde{p}(\bf{z}^{(l)})/\tilde{q}(\bf{z}^{(l)})$ and hence
\begin{equation}
\boxed{	\mathbb{E}[f] \simeq \sum_{l=1}^L w_l f(\bf{z}^{(l)})}
\end{equation}
where we have defined
\begin{equation}
	w_l = \frac{\tilde{r}_l}{\sum_m \tilde{r}_m} = \frac{\tilde{p}(\bf{z}^{(l)})/q(\bf{z}^{(l)})}{\sum_m \tilde{p}(\bf{z}^{(m)})/q(\bf{z}^{(m)})}.
\end{equation}

A major drawback of the importance sampling method is the potential to produce results that are arbitrarily in error and with no diagnostic indication. This also highlights a key requirement for the sampling distribution $q(z)$, namely that it should not be small or zero in regions where $p(z)$ may be significant.


#### 11.1.5 Sampling-importance-resampling
As in the case of rejection sampling, the **sampling-importance-resampling** (SIR) approach also makes use of a sampling distribution $q(\bf{z})$ but avoids having to determine the constant $k$ as required in rejection sampling. There are two stages to the scheme. In the first stage, $L$ samples $\bf{z}^{(1)},\cdots,\bf{z}^{(L)}$ are drawn from $q(\bf{z})$. Then in the second stage, weights $w_1,\cdots,w_L$ are constructed using the same way as in importance sampling. Finally, a second set of $L$ samples is drawn from the discrete distribution $(\bf{z}^{(1)},\cdots,\bf{z}^{(L)})$ with probabilities given by the weights $(w_1,\cdots,w_L)$. The resulting $L$ samples are only approximately distributed according to $p(\bf{z})$, but the distribution becomes correct in the limit $L \rightarrow \infty$.

#### 11.1.6 Sampling and the EM algorithm
Remain to be written. 

### 11.2 Markov Chain Monte Carlo
The sampling methods we have discussed suffer from severe limitations in the high dimensional spaces. In this section we move on to a very general and powerful framework called **Markov chain Monte Carlo** (MCMC), which allows sampling from a large class of distributions, and which scales well with the dimensionality of the sample space.

#### 11.2.1 The basic Metropolis algorithm
In the basic Mrteopolis algorithm, we assume that the proposal distribution is symmetric, which means
\begin{equation}
q(z_A|z_B) = q(z_B|z_A)
\end{equation} holds for all choices of $z_A$ and $z_B$. 
The candidate sample is then accepted with probability
\begin{equation}
A(z^*, z^{\tau}) = min(1, \frac{\tilde{p(z^*)}}{\tilde{p(z^{\tau})}})
\end{equation}
And the process is as following. We first choose a random number $u$ with uniform distribution over the unit interval (0,1) and then accepting the sample if $A(z ,z(\tau)) > u$. Note that if the step from $z(\tau)$ to $z$ causes an increase in the value of $p(z)$, then the candidate point is certain to be kept.


Note that different from the rejection sampling, when a candidate point is rejected, in the basic Metropolis algorithm, the previous sample is included instead in the final list of samples, leading to multiple copies of samples. And the sampled data are not independent from each other. 
![屏幕快照 2019-03-12 下午3.32.01.png](resources/9AFA42341956012D1B8C6B5DED227EF2.png =568x324)
After $\tau$ steps, the random walk has only travelled a distance that on average is proportional to the square root of $\tau$. This square root dependence is typical of random walk behaviour and shows that random walks are very inefficient in exploring the state space.

#### 11.2.2 Markov chains
A first-order Markov chain is defined to be a series of random variables $z(1),\dots,z(M)$ such that the following conditional independence property holds for $m \in {1, . . . , M − 1}$
\begin{equation}
p(z^{m+1}|z_1, \dots, z_m) = p(z^{m+1}|z^m)
\end{equation}
which can be represented as a directed graph:
![屏幕快照 2019-03-12 下午3.40.15.png](resources/C6CB6866E6A614D78BE37706991F0B6C.png =713x145)


For a homogeneous Markov chain with transition probabilities $T(\bf{z}',\bf{z})$, the distribution $p^*(\bf{z})$ is **invariant** if
\begin{equation}
	p^*(\bf{z}) = \sum_{\bf{z}'} T(\bf{z}',\bf{z})p^*(\bf{z'}).
\end{equation}


A sufficient (but not necessary) condition for ensuring that the required distribution $p(\bf{z})$ is invariant is to choose the transition probabilities to satisfy the property of **detailed balance**, defined by
\begin{equation}
	p^*(\bf{z})T(\bf{z,z'}) = p^*(\bf{z'})T(\bf{z',z})
\end{equation}
A Markov chain that respects detailed balance is said to be **reversible**.


If for $m \rightarrow \infty$, the distribution $p(\bf{z}^{(m)})$ converges to the required invariant distribution $p^*(\bf{z})$, irrespective of the choice of initial distribution $p(\bf{z}^{(0)})$, we call this Markov chain to be \imp{ergodic} and the corresponding invariant distribution to be an \imp{equilibrium} distribution. Clearly, an ergodic Markov chain can have only one equilibrium distribution. It can be shown that a homogeneous Markov chain will be ergodic, subject only to weak restrictions on the invariant distribution and the transition probabilities <cite>(Neal, 1993)</cite>.


In practice we often construct the transition probabilities from a set of ‘base’
transitions $B_1 ,\dots, B_K$ . This can be achieved through a mixture distribution of the form
\begin{equation}
T(z,z') = \sum_{k=1}^K\alpha_kB_k(z', z)
\end{equation}
And the base transitions can also be combined as:
\begin{equation}
T(z', z) = \sum_{z_1}\dots\sum_{z_{K-1}}B_1(z',z_1)\dots B_{K-1}(z_{K-2}, z_{K-1})B_K(z_{K-1},z_K)
\end{equation}

#### 11.2.3 The Metropolis-Hastings algorithm
The Metropolis-Hastings algorithm is a generation of the basic Metropolis algorithm, where the proposal distribution is no longer a symmetric function of its arguments. At step $\tau$ of the algorithm, in which the current state is $\bf{z}^{(\tau)}$, we draw a sample $\bf{z}^*$ from the distribution $q_k(\bf{z}|\bf{z}^{(\tau)})$, which is the proposal distribution, and then accept it with probability $A_k(\bf{z}^*,\bf{z}^{(\tau)})$ where
\begin{equation}
	A_k(\bf{z}^*,\bf{z}^{(\tau)}) = \min \left(  1, \frac{\tilde{p}(\bf{z}^*) q_k(\bf{z}^{(\tau)}|\bf{z}^*)}{\tilde{p}(\bf{z}^{(\tau)})q_k(\bf{z}^*|\bf{z}^{(\tau)})} \right).
\end{equation}
Here $k$ labels the members of the set of possible transitions being considered. However, if the length scales over which the distributions vary are very different in different directions, then the Metropolis Hastings algorithm can have very slow convergence. 

### 11.3 Gibbs Sampling
Gibbs sampling can be viewed as a special case of the Metropolis-Hastings algorithm. 


**Gibbs sampling.** Suppose we want to sample from the distribution $p(\bf{z}) = p(z_1,\cdots,z_M)$, we have the following sampling procedure called Gibbs sampling.
![屏幕快照 2019-03-12 下午5.37.35.png](resources/EB80D9C81A5F11682C404445F7D0DEFE.png =476x266)
A sufficient condition for ergodicity is that none of the conditional distributions be anywhere zero. If this is the case, then any point in $z$ space can be reached from any other point in a finite number of steps involving one update of each of the component variables. If this requirement is not satisfied, so that some of the conditional distributions have zeros, then ergodicity, if it applies, must be proven explicitly.


We can obtain the Gibbs sampling procedure as a particular instance of the Metropolis-Hastings algorithm as follows. Similar to the Metropolis-Hastings algorithm, the acceptance probability is given by:
\begin{equation}
	A(\bf{z}^*,\bf{z})=\frac{p(\bf{z}^*)q_k(\bf{z|z}^*)}{p(\bf{z})q_k(\bf{z}^*|\bf{z})}=\frac{p(z_k^*|\bf{z}_{\backslash k}^*)p(\bf{z}_{\backslash k}^*)p(z_k|\bf{z}_{\backslash k}^*)}{p(z_k|\bf{z}_{\backslash k})p(\bf{z}_{\backslash k})p(z_k^*|\bf{z}_{\backslash k})} = 1
\end{equation} where we should note that $\bf{z}_{\backslash k} = \bf{z}_{\backslash k}^*$


### 11.4 Slice Sampling
One of the difficulties with the Metropolis algorithm is the sensitivity to step size. If this is too small, the result is slow decorrelation due to random walk behaviour, whereas if it is too large the result is inefficiency due to a high rejection rate. The technique of **slice sampling** <cite>(Neal, 2003)</cite>> provides an adaptive step size that is automatically adjusted to match the characteristics of the distribution. Again it requires that we are able to evaluate the unnormalized distributions $\tilde{p}{(\bf{z})}$.


Consider only the univariate case. The motivation here is to sample uniformly from the area under the distribution given by
\begin{equation}
	\hat{p}(z,u) = \begin{cases}
		1/Z_p \qquad &\text{if $0 \leq u \leq \tilde{p}(z)$}\\
		0 \qquad &\text{otherwise}
	\end{cases}
\end{equation}
where $Z_p = \int \tilde{p}(z) d z$. The marginal distribution over $z$ is given by
\begin{equation}
	\int \hat{p}(z,u) d u = \int_{0}^{\tilde{p}(z)} \frac{1}{Z_p} d u = \frac{\tilde{p}(z)}{Z_p} = p(z)
\end{equation}
We can sample from $p(z)$ by sampling from $\hat{p}(z,u)$ and then ignoring the $u$ values. This can be achieved by alternately sampling $z$ and $u$. Given the value of $z$ we evaluate $\tilde{p}(z)$ and then sample $u$ uniformly in the range $0 \leq u \leq \tilde{p}(z)$, which is straightforward. Then we fix $u$ and sample $z$ uniformly from the ``slice'' through the distribution defined by $\{ z: \tilde{p}(z) > u \}$.


Slice sampling can be applied to multivariate distributions by repeatedly sam- pling each variable in turn, in the manner of Gibbs sampling. This requires that we are able to compute, for each component $z_i$, a function that is proportional to p(zi |z\i ).

### 11.5 The Hybrid Monte Carlo Algorithm
One of the major limitations of the Metropolis algorithm is that it can exhibit random walk behaviour whereby the distance traversed through the state space grows only as the square root of the number of steps.


The Hybrid Monte Carlo algorithm is based on an analogy with physical systems and can make large changes to the system state while keeping the rejection probability small.

#### 11.5.1 Dynamical systems
The dynamical approach to stochastic sampling origins from simulating the behaviour of physical systems evolving under Hamiltonian dynamics. 


The energy of the whole system(Hamiltonian function) is defined as:
\begin{equation}
H(z,r) = E(z) + K(r)
\end{equation} where $E(z)$ denotes the potential energy of the system when in state $z$ and $K(r) = \frac{1}{2}|r|^2 = \frac{1}{2}\sum_r r_i^2$, which denotes the kinetic energy. We can now express the dynamics of the system in terms of the Hamiltonian equations given by:
\begin{aligned}
\frac{dz_i}{d\tau} &= \frac{\partial{H}}{\partial{r_i}} \\
\frac{dr_i}{d\tau} &= -\frac{\partial{H}}{\partial{z_i}}
\end{aligned}
Note that during the evolution of this dynamical system, the value of the Hamiltonian $H$ is constant. A second important property of Hamiltonian dynamical systems, known as Liouville’s Theorem, is that they preserve volume in phase space.


Now consider the joint distribution over phase space whose total energy is the Hamiltonian, i.e., the distribution given by
\begin{equation}
p(z,r) = \frac{1}{Z_H}\exp(-H(z,r))
\end{equation}
Though $H$ is constant, $z$ and $r$ is changing in the way which controls the $H$ to be a constant. 


 In order to arrive at an ergodic sampling scheme, we can introduce additional moves in phase space that change the value of H while also leaving the distribution $p(z,r)$ invariant. **The leapfrog method** is a useful numerical scheme to solve the Hamilton dynamical equations, which preserves the volume exactly and works as follows:
\begin{align}
	p_i(t+\epsilon/2) &= p_i(t) - (\epsilon/2)\frac{\partial E}{\partial q_i} (q(t)) \\
	q_i(t+ \epsilon) &= q_i(t) + \epsilon \frac{p_i(t+\epsilon/2)}{m_i} \\
	p_i(t+\epsilon) &= p_i(t+\epsilon/2) - (\epsilon/2)\frac{\partial E}{\partial q_i}(q(t+\epsilon))
\end{align}


To sum up, the Hamiltonian dynamical approach involves alternating between a series of leapfrog updates and a resampling of the momentum variables from their marginal distribution.


#### 11.5.2 Hybrid Monte Carlo(referred to <cite>Yang Song</cite>)
Each iteration of the HMC algorithm has two steps. The first only changes the momentum; the second may change both position and momentum. Both steps leave the canonical joint distribution of $(z,r)$ invariant, and hence
their combination also leaves this distribution invariant.


In the first step, new values for the momentum variables are randomly drawn from their Gaussian distribution, independently of the current values of the position variables. Since $z$ is not changed, and $r$ is drawn from its correct conditional distribution given $z$ (the same as its marginal distribution, due to independence), this step obviously leaves the canonical joint distribution invariant as a result of the similar argument in Gibbs sampling algorithm.


In the second step, a Metropolis update is performed, using Hamiltonian dynamics to propose a new state. Starting with the current state, $(z, r)$, Hamiltonian dynamics is simulated for $L$ steps using the leapfrog method (or some other reversible method that preserves volume), with a stepsize of $\varepsilon$. Here, $L$ and $\varepsilon$ are \textbf{parameters} of the algorithm, which need to be
tuned to obtain good performance. The momentum variables at the end of this $L$-step trajectory are then \textbf{negated}, giving a proposed state $(q^*,p^*)$. This proposed state is accepted as the next state of the Markov chain with probability
\begin{equation}
	\min \left[ 1, \exp(- H(q^*,p^*) + H(q,p)) \right]
\end{equation}
If the proposed state is not accepted (i.e., it is rejected), the next state is the same as the current state (and is counted again when estimating the expectation of some function of state by its average over states of the Markov chain).The negation of the momentum variables at the end of the trajectory makes the Metropolis proposal symmetrical (in order to be reversible), as needed for the acceptance probability above to be valid. This negation need not be done in practice, since $K(r) = K(-r)$, and the momentum will be replaced before it is used again, in the first step of the next iteration. 

ps: There are much more to study in the MCMC. Remain to be written. 

## 12. Continuous Latent Models
In this section, we explore models in which some, or all of the latent variables are continuous. 

### 12.1 Principal component analysis
There are mainly two definitions of PCA which result in the same algorithm. PCA can be defined as the orthogonal projection of the data onto a lower dimensional linear space, known as principal space,  such that the variance of the projected data is maximized. Equivalently, it can be defined as the linear projection that minimize the average projection cost, defined as the mean squared distance between the data points and their projections. 

#### 12.1.1 Maximum variance formulation
\begin{aligned}
\widetilde{x} &= \frac{1}{N}\sum_{n=1}^N x_n \\
\sigma^2 &= \frac{1}{N}\sum_{n=1}^N\{u_1^{\intercal}x_n - u_1^{\intercal}\widetilde{x}\}^2 \\
&= u_1^{\intercal}Su_1
\end{aligned} where $S$ is the covariance matrix defined by
\begin{equation}
S = \frac{1}{N}\sum_{n=1}^N(x_n - \widetilde{x})(x_n - \widetilde{x})^{\intercal}
\end{equation}
Then we maximize the following:
\begin{equation}
u_1^{\intercal}Su_1 + \lambda_1(1-u_1^{\intercal}u_1)
\end{equation} which gives:
\begin{equation}
Su_1 = \lambda_1 u_1
\end{equation} by letting the derivative over $u_1$ to be 0. We have found that  $u_1$ must be an eigenvector of $S$, which gies:
\begin{equation}
u_1^{\intercal}Su_1 = \lambda_1
\end{equation}


In general, principal component analysis involves evaluating the mean $\widetilde{x}$ and the covariance matrix $S$ of the data set and then finding the $M$ eigenvectors of $S$ corresponding to the $M$ largest eigenvalues.


#### 12.1.2 Minimum-error formulation
We shall use the squared distortion and our goal is to minimize
\begin{equation}
	J = \frac{1}{N} \sum_{n=1}^N ||\bf{x}_n-\tilde{\bf{x}}_n||^2.
\end{equation}
After some calculation, we can get the same result as the maximum variance formulation. 

#### 12.1.3 Whitening
Another name for the method is called \imp{sphereing}. This is a substantial normalization of the data to give it \textbf{zero mean} and \textbf{unit covariance}, inspired by PCA. To do this, we first diagonalize the data covariance $\bf{S}$ to give
\begin{equation}
	\bf{SU} = \bf{UL}.
\end{equation}
Then we define, for each data point $\bf{x}_n$, a transformed value given by
\begin{equation}
	\bf{y}_n = \bf{L}^{1/2}\bf{U}^{\intercal} (\bf{x}_n - \bar{\bf{x}}).
\end{equation}


#### 12.1.4 PCA for high-dimensional data
We may use PCA for high-dimensional data, where the dimension of the data is even more than the size of the dataset. In such case, we would like to search for an equivalent method to perform the PCA efficiently. 


First, we define $\bf{X}$ as the $(N \times D)$-dimensional centered data matrix, whose $n^{th}$ row is given by $(\bf{x}_n-\bar{\bf{x}})^{\intercal}$. The covariance matrix can then be written as $\bf{S} = N^{-1} \bf{X}^{\intercal} \bf{X}$, and the corresponding eigenvector equation becomes
\begin{equation}
	\frac{1}{N} \bf{X}^{\intercal} \bf{X}\bf{u}_i = \lambda_i \bf{u}_i.
\end{equation}
Now pre-multiply both sides by $\bf{X}$ to give
\begin{align}
	\frac{1}{N} \bf{XX}^{\intercal} (\bf{Xu}_i) &= \lambda_i (\bf{Xu}_i) \\
	\frac{1}{N} \bf{XX}^{\intercal} \bf{v}_i &= \lambda_i \bf{v}_i 
\end{align} where we define $v_i = \bf{X}u_i$. We can see that $v_i$ is the eigenvector of the $XX^{\intercal}$ whose size is $NxN$ instead of $DxD$. We use a trick to lower the dimension. 


In general, we firstly evaluate the $XX^{\intercal}$ and then find its eigenvectors and eigenvalues and then compute the eigenvectors in the original data space using
\begin{equation}
u_i = \frac{1}{(N\lambda_i)^{0.5}}X^{\intercal}v_i
\end{equation}


### 12.2 Probabilistic PCA
We here show that PCA can also be expressed as the maximum likelihood solution of a probabilistic latent variable model, which is called the probabilistic PCA. 
* Probabilistic PCA represents a constrained form of the Gaussian distribution in which the number of free parameters can be restricted while still allowing the model to capture the dominant correlations in a data set.
* We can derive an EM algorithm for PCA that is computationally efficient in situations where only a few leading eigenvectors are required and that avoids having to evaluate the data covariance matrix as an intermediate step.
* The combination of aprobabilistic model and EM allows us to deal with missing values in the data set.
* Mixtures of probabilistic PCA models can be formulated in a principled way and trained using the EM algorithm.
* Probabilistic PCA forms the basis for a Bayesian treatment of PCA in which the dimensionality of the principal subspace can be found automatically from the data.
* The existence of a likelihood function allows direct comparison with other probabilistic density models. By contrast, conventional PCA will assign a low reconstruction cost to data points that are close to the principal subspace even if they lie arbitrarily far from the training data.
* Probabilistic PCA can be used to model class-conditional densities and hence be applied to classification problems.
* The probabilistic PCA model can be run generatively to provide samples from the distribution.

#### 12.2.1 Model
The prior over $\bf{z}$ is given by (a more general Gaussian only leads to the same probabilistic model)
\begin{equation}
	p(\bf{z}) = \cal{N}(\bf{z}|\bf{0},\bf{I}).
\end{equation}
The conditional distribution of the observed variable $\bf{x}$ is given by
\begin{equation}
	p(\bf{x|z}) = \cal{N}(\bf{x}|\bf{Wz}+\boldsymbol{\mu},\sigma^2\bf{I}).
\end{equation}
where $\bf{W}$ is a $D \times M$ matrix. Note that this factorizes with respect to the elements of $\bf{x}$ and is hence an example of the **naive Bayes model**.
\begin{equation}
p(x) = \int p(x|z)p(z)dz
\end{equation}
Since it is a linear Gaussian model, we have
\begin{equation}
p(x) = \cal{N}(x|\mu,C)
\end{equation} where $C = WW^{\intercal}+\sigma^2I$


In addition, the posterior distribution over $\bf{z}$ is
\begin{equation}
p(z|x) = \cal{N}(z|M^{-1}W^{\intercal}(x-\mu), \sigma^2M^{-1})
\end{equation}
We can also use a directed graph to represent the probabilistic PCA as following:
![屏幕快照 2019-03-14 下午4.23.01.png](resources/93E647C2641E7463491E87BBEB74A0FC.png =575x190)

#### 12.2.2 Maximum likelihood PCA
Now we consider how to determine the model parameters using maximum likelihood. 
The log likelihood function is given by
\begin{aligned}
	\ln p(\bf{X}|\boldsymbol{\mu},\bf{W},\sigma^2) &= \sum_{n=1}^N lnp(x_n|W,\mu,\sigma^2) \\
	&= \frac{-ND}{2}\ln(2\pi)-\frac{N}{2}\ln|\bf{C}|-\frac{1}{2}\sum_{n=1}^N (\bf{x}_n-\boldsymbol{\mu})^{\intercal} \bf{C}^{-1} (\bf{x}_n - \boldsymbol{\mu}).
\end{aligned}

Setting the derivatives with respect to $\boldsymbol{\mu}$, $\bf{W}$ and $\sigma^2$ to zero, we obtain
\begin{align}
	\bf{\mu} &= \bar{\bf{x}} \\
	\bf{W}_{ML} &= \bf{U}_M (\bf{L}_M-\sigma^2 \bf{I})^{1/2}\bf{R} \\
	\sigma_{ML}^2 &= \frac{1}{D-M} \sum_{i=M+1}^D \lambda_i
\end{align} where $U_M$ is a $D x M$ matrix whose columns are given by any subset (of size $M$) of the eigenvectors of the data covariance matrix $S$. In addition,  Tipping and Bishop (1999b) showed that the maximum of the likelihood function is obtained when the $M$ eigenvectors are chosen to be those whose eigenvalues are the $M$ largest (all other solutions being saddle points). 


#### 12.2.3 EM algorithm for PCA
The complete-data log likelihood function takes the form
\begin{equation}
	\ln p(\bf{X,Z}|\boldsymbol{\mu},\bf{W},\sigma^2) = \sum_{n=1}^N \{ \ln p(\bf{x}_n|\bf{z}_n)+ \ln p(\bf{z}_n) \}
\end{equation}

**The E step**. 
\begin{align}
\mathbb{E}[\ln p(\bf{X,Z}|\boldsymbol{\mu},\bf{W},\sigma^2)] = -\sum_{n=1}^N \{ \frac{D}{2}\ln (2\pi\sigma^2)+\frac{1}{2}{Tr}(\mathbb{E}[\bf{z}_n\bf{z}_n^{\intercal}])\\ +\frac{1}{2\sigma^2}||\bf{x}_n-\boldsymbol{\mu}||^2-\frac{1}{\sigma^2}\mathbb{E}[\bf{z}_n]^{\intercal} \bf{W}^{\intercal}(\bf{x}_n-\boldsymbol{\mu})\\+\frac{M}{2}\ln (2\pi)+\frac{1}{2\sigma^2}{Tr}(\mathbb{E}[\bf{z}_n\bf{z}_n^{\intercal}]\bf{W}^{\intercal}\bf{W}) \}
\end{align}
We use the old parameter values to evaluate
\begin{align}
	\mathbb{E}[\bf{z}_n] &= \bf{M}^{-1} \bf{W}^{\intercal} (\bf{x}_n-\bar{\bf{x}}) \\
	\mathbb{E}[\bf{z}_n\bf{z}_n^{\intercal}] &= \sigma^2 \bf{M}^{-1} + \mathbb{E}[\bf{z}_n] \mathbb{E}[\bf{z}_n]^{\intercal}.
\end{align}

**The M step**. We maximize with respect to $\bf{W}$ and $\sigma^2$, keeping the posterior statistics fixed. The M-step equations are
\begin{align}
	\bf{W}_{new} &= \left[ \sum_{n=1}^N (\bf{x}_n-\bar{\bf{x}})\mathbb{E}[\bf{z}_n]^{\intercal} \right]\left[
	\sum_{n=1}^N \mathbb{E}[\bf{z}_n\bf{z}_n^{\intercal}]\right]^{-1} \\
	\sigma_{new}^2 &= \frac{1}{ND} \sum_{n=1}^N \{ ||\bf{x}_n-\bar{\bf{x}}||^2 - 2\mathbb{E}[\bf{z}_n]^{\intercal} \bf{W}_{new}^{\intercal}(\bf{x}_n-\bar{\bf{x}})\\
	&\quad + {Tr}(\mathbb{E}[\bf{z}_n\bf{z}_n^{\intercal}]\bf{W}_{new}^{\intercal}\bf{W}_{new}) \}.
\end{align}



### 12.3 Bayesian PCA
We introduce the prior distribution of $W$ to give the Bayesian PCA.
![屏幕快照 2019-03-14 下午5.27.35.png](resources/6298F130F8E200CFC7698AFD767EA0AF.png =572x192)
It can prune the surplus dimensions in the principal subspace out of the model. This corresponds to an example of **automatic relevance determination**, or **ARD**. Specifically, we define an independent Gaussian prior over each column of $\bf{W}$. Each Gaussian has an independent variance governed by a precision hyperparameter $\alpha_i$ so that
\begin{equation}
	p(\bf{W}|\boldsymbol{\alpha}) = \prod_{i=1}^M \left( \frac{\alpha_i}{2\pi} \right)^{D/2} \exp\left\{ -\frac{1}{2}\alpha_i \bf{w}_i^{\intercal} \bf{w}_i \right\}
\end{equation}
where $\bf{w}_i$ is the $i^{{th}}$ column of $\bf{W}$. Note that we define an independent Gaussian prior over each column of $W$.  


The marginal likelihood is given by
\begin{equation}
	p(\bf{X}|\boldsymbol{\alpha,\mu},\sigma^2) = \int p(\bf{X|W},\boldsymbol{\mu},\sigma^2) p(\bf{W}|\boldsymbol{\alpha}) d \bf{W}.
\end{equation}
We them make use of the Laplace approximation to evaluate this integral under the assumption that the posterior distribution is sharply peeked, as will occur for sufficiently \textbf{large} data sets. By analogy to what we discussed in Section \ref{DEigens}, the marginal likelihood with respect to $\alpha_i$ take the simple form
\begin{equation}
	\alpha_i^{new} = \frac{D}{\bf{w^*}_i^{\intercal} \bf{w^*}_i}
\end{equation}

As a result, we need to evaluate the **optimal** $\bf{W}^*$ in every **iteration**. This can been achieved by a variant of EM algorithm specially modified to evaluate MAP results. The re-estimation of $\alpha_i$ and $\bf{W}$ are interleaved with each other. The E-step equations are again given by the results in probabilistic PCA. However, since the EM has been modified for MAP results, the M-step should be changed correspondingly. The M-step equation for $\sigma^2$ is again given by the result in the probabilistic. But the M-step equation for $\bf{W}$ is modified to give
\begin{equation}
	\bf{W}_{new} = \left[ \sum_{n=1}^N (\bf{x}_n-\bar{\bf{x}})\mathbb{E}[\bf{z}_n]^{\intercal} \right] \left[ \sum_{n=1}^N \mathbb{E}[\bf{z}_n\bf{z}_n^{\intercal}] +\sigma^2 \bf{A}\right]^{-1}
\end{equation}
where $\bf{A} = {diag}(\alpha_i)$. The value of $\boldsymbol{\mu}$ is given by the sample mean as before.


In general, the values for $\alpha_i$ will be found iteratively by maximizing the marginal likelihood function in which $W$ has been integrated out. As a result of this optimization, some of the $\alpha_i$ may be driven to infinity, with the corresponding parameters vector $W_i$ being driven to zero (the posterior distribution becomes a delta function at the origin) giving a sparse solution. The effective dimensionality of the principal subspace is then determined by the number of finite $W_i$ values, and the corresponding vectors $W_i$ can be thought of as 'relevant' for modelling the data distribution. In this way, the Bayesian approach is automatically making the trade-off between improving the fit to the data, by using a larger number of vectors $W_i$ with their corresponding eigenvalues Ai each tuned to the data, and reducing the complexity of the model by suppressing some of the $W_i$ vectors. The origins of this sparsity were discussed earlier in the context of relevance vector machines. It is quite similar to **RVM**. 

### 12.4 Factor analysis



### 12.5 kernel PCA
Here we bring kernel trick into the PCA and get a nonlinear generalization called **kernel PCA**.


Suppose that we have already subtracted the sample mean from each of the vectors $\bf{x}_i$ and the principal components are defined by the eigenvectors $u_i$ of the covariance matrix
\begin{equation}
\bf{S}u_i = \lambda_iu_i
\end{equation} where we have
\begin{equation}
S = \frac{1}{N}\sum_{n=1}^N x_nx_n^{\intercal}
\end{equation}
Now consider a nonlinear transformation $\boldsymbol{\phi}(\bf{x})$ into an $M$-dimensional feature space, so that each data point $\bf{x}_n$ is thereby projected onto a point $\boldsymbol{\phi}(\bf{x}_n)$. We can now perform standard PCA in the feature space analogously:
\begin{equation}
	\bf{Cv}_i = \frac{1}{N}\sum_{n=1}^N \boldsymbol{\phi}(\bf{x}_n)\boldsymbol{\phi}(\bf{x}_n)^{\intercal} \bf{v}_i = \lambda_i \bf{v}_i 
\end{equation}
If we combine the $\phi(x_n)^{\intercal}$ and $v_i$ together, we have:
\begin{align}
v_i &= \frac{1}{N\lambda_i}\sum_{n=1}^N \phi(x_n)\{\phi(x_n)^{\intercal}v_i\} \\
&= \sum_{n=1}^N a_{in}\phi(x_n)
\end{align}
Then we have
\begin{equation}
\frac{1}{N}\sum_{n=1}^N \phi(x_n)\phi(x_n)^{\intercal}\sum_{m=1}^Na_{im}\phi(x_m) = \lambda_i \sum_{n=1}^N a_{in}\phi(x_n)
\end{equation}
The key step is now to express this in terms of the kernel function $k(x_n,x_m) = \phi(x_n)^{\intercal}\phi(x_m)$, for which we multiply both sides by $\phi(x_l)^{\intercal}$.
\begin{equation}
\frac{1}{N}\sum_{n=1}^Nk(x_L, x_n)\sum_{m=1}^Na_{im}k(x_n, x_m) = \lambda_i \sum_{n=1}^N a_{in}k(x_l, x_n)
\end{equation}
\begin{equation}
\bf{K}^2 \bf{a}_i = \lambda_i N \bf{Ka}_i
\end{equation}
This can be proved to be equivalent to
\begin{equation}
	\bf{Ka}_i = \lambda_i N \bf{a}_i
\end{equation}
for the purpose of predictions. One more thing to do is to normalize $\bf{v}_i$, which can be done in the form of $\bf{a}_i$ by
\begin{equation}
	1 = \bf{v}_i^{\intercal} \bf{v}_i = \bf{a}_i^{\intercal} \bf{Ka}_i = \lambda_i N \bf{a}_i^{\intercal}\bf{a}_i.
\end{equation}
The projection of a point $x$ onto eigenvector $i$ is given by
\begin{align}
y_i(x) &= \phi(x)^{\intercal}v_i \\
    &= \sum_{n=1}^Na_{in}\phi(x)^{\intercal}\phi(x_n) \\
    &= \sum_{n=1}^Na_{in}k(x,x_n)
\end{align}



So far we have assumed that the projected data set given by $\phi(x_n)$ has zero mean, which in general will not be the case. We cannot simply compute and then subtract off the mean, since we wish to avoid working directly in feature space. We woulk like to find a simple way to handle the $non-zero$ means problem. The projected data points after **centralizing** are given by
\begin{equation}
	\tilde{\boldsymbol{\phi}}(\bf{x}_n) = \boldsymbol{\phi}(\bf{x}_n) - \frac{1}{N} \sum_{l=1}^N \boldsymbol{\phi}(\bf{x}_l)
\end{equation}
and the corresponding elements of the Gram matrix are given by
\begin{align}
	\tilde{\bf{K}} = \bf{K} - \bf{1}_N\bf{K} - \bf{K} \bf{1}_N + \bf{1}_N \bf{K} \bf{1}_N
\end{align} where $\bf{1}_N$ denotes the $N \times N$ matrix in which every element takes the value $1/N$. After having established the relationship between $\tilde{\bf{K}}$ and $\bf{K}$, the more generalized problem is solved by direct analogy.


### 12.6 Nonlinear Latent Variable Models

#### 12.6.1 Independent component analysis(ICA)
We begin by considering models in which the observed variables are related linearly to the latent variables, but for which the latent distribution is non-Gaussian. This method arises when we consider a distribution over the latent variables that factorizes, so that
\begin{equation}
p(z) = \prod_{i=1}^Mp(z_i)
\end{equation}

#### 12.6.2 Autoassociative neural networks
![屏幕快照 2019-03-15 下午3.23.25.png](resources/F84DBD979CB29E5DF6AC25B4DB69432C.png =586x187)
![屏幕快照 2019-03-15 下午3.23.33.png](resources/4ACA0DF8C512204D47F9A4DDC7CC64FD.png =588x252)
I think it is quite similar to the encoder-decoder structure. 

## 13. Sequential Data

### 13.1 Markov Models
![屏幕快照 2019-03-15 下午3.31.15.png](resources/19B9D433D75B61F09FABB26C05160EE8.png =582x128)
The model is defined as:
\begin{equation}
p(x_1, \dots, x_N) = p(x_1)\prod_{n=2}^Np(x_n|x_{n-1})
\end{equation}