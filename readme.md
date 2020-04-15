A request has been put forth to reproduce the results of  ["Dropout: A Simple Way to Prevent Neural Networks from Overitting" by Srivastava, Hinton, Krizhevsky, Sutskever and Salakhutdinov (2014)](http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf) , table 10, see below.  
![table 10 of (Srivastava, Hinton)](https://drive.google.com/uc?id=1LX9quIO5OFw6TPBKXzuvaH9z1M2cSJ0T)

This blogpost will describe the steps taken towards that reproduction, implemented using both PyTorch and Keras, and discuss the results. Before any experiments are done, we will provide information regarding the two datasets, what dropout is, the two different types of dropout used here and the two different architectures used; as can be observed in table 10.

## Datasets
Table 10 treats two different datasets: MNIST, and CIFAR-10. 

### CIFAR-10 Dataset

CIFAR-10, from [Kaggle](https://www.kaggle.com/c/cifar-10):<br>
"CIFAR-10  is an established computer-vision dataset used for object recognition. It is a subset of the 80 million tiny images dataset and consists of 60,000 32x32 color images containing one of 10 object classes, with 6000 images per class. It was collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton."

![CIFAR10](http://pytorch.org/tutorials/_images/cifar10.png)

As can be seen in the image above, the 10 classes are: ‘airplane’, ‘automobile’, ‘bird’, ‘cat’, ‘deer’,
‘dog’, ‘frog’, ‘horse’, ‘ship’, ‘truck’. The images in CIFAR-10 are of
size 3x32x32, i.e. 3-channel color images of 32x32 pixels in size.



### MNIST Dataset

MNIST, from [Kaggle](https://www.kaggle.com/c/digit-recognizer): <br>
"MNIST ("Modified National Institute of Standards and Technology") is the de facto “hello world” dataset of computer vision. Since its release in 1999, this classic dataset of handwritten images has served as the basis for benchmarking classification algorithms. As new machine learning techniques emerge, MNIST remains a reliable resource for researchers and learners alike."

Also includes 10 classes: '0', ..., '9'. The images in MNIST are of size 1x28x28, i.e. 1-channel B-W images of 28x28 pixels in size.

<a title="By Josef Steppan [CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0)], from Wikimedia Commons" href="https://commons.wikimedia.org/wiki/File:MnistExamples.png"><img width="512" alt="MnistExamples" src="https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png"/></a>

## What is dropout?
For those that are not familiar yet with the term and usage of "dropout" in neural networks: dropout is a technique used as a measure against overfitting within (deep) neural networks. Just like merging multiple networks into one almost always increases the generalization capability of the resulting network (networks in example have different architectures, find different (local) optima or find them in a different way; and so merging retains information of both), dropout imitates this beneficial pro-generalization behaviour without the need to actually train separate networks. With dropout each node and its connections has a chance to be withold of training during a single training step, effectively each time training a 'different' network.s As such specialistic (thus prone to overfitting) co-adaptations are less likely to form; we want each node to have a meaningfull contribution on its own, and not to be compensating- or be dependent on another node. For further information we refer to the paper wherefrom we will try to reproduce results of (Srivastava et al). 

## Difference between Bernouilli and Gaussian dropout
In table 10 one might notice the distinction between two types of dropout: Bernouilli dropout and Gaussian dropout. This refers to the distribution over which we sample the weights that mathmetically effectuates the 'dropout' of a node. With a Bernouilli distribution we have chance p that this weight will be 1 (and the the node will be fully retained during the trainig step) and chance 1-p for weight 0 (the node will be disregarded during the training step). When we sample these weights from a Gaussian distribution, but in general and for this paper as well the special case of a normal distribution $N(\mu,\sigma^2)\rightarrow N(0,1)$, the weights can be a range of values instead of only one and zero, with each value having a different chance at occuring. For a slightly more mathmetical explanation, please scroll to <a href="#name_of_target">"The Bernoulli and Gaussian Dropout Class"</a>

## A priori expectations 
Already before doing any experiments we should note the statistical relationship between the Bernoulli and Gaussian distribtution and therewith what its preliminary implication is on our test results. Through the central limit theorem, when we sample an infinite amount of 'batches' from a Bernouilli distribution we will obtain a binomial distribtution. When we sample with p = 0.5 and thus 1-p = 0.5, our expected value lies in the middle of our domain [0,1] with $E[x]=\frac{1}{2}\cdot 0 + \frac{1}{2}\cdot 1 = \frac{1}{2}$. As such, this binomial becomes symmetric and we can convert it to a Gaussian distribution. So, when we perform Bernouilli dropout an infinite amount of times and each time record the amount of leftover nodes in a histogram, we retrieve a Gaussian. Preferably we have an infinite amount of data and computing power, then Bernouilli would converge to Gaussian. As we know this will naturally happen given a Bernouilli, why don't we sample from a Gaussian in the first place, even if (or rather especially if) we have less than infinite data and computing power? We know it to be the natural convergence behaviour that furthermore occurs in the real-world as well! For that reason we presume that a Gaussian approach should perform at least as well as a Bernoulli approach. Indeed when we look at table 10 this is the case, supporting this conjecture.

## Used network architectures
Lastly we consider the neural network architectures used in table 10 of the paper. For the MNIST dataset, 2 (hidden) layers with 1024 units each are used in a feedforward neural network. One might note that besides this, for the CIFAR-10 dataset three additional convolutional layers have been used. The addition originates from the nature of the datasets. The MNIST dataset already is very 'feature'-istic, while the CIFAR-10 dataset is not. For the authors to better compare the effect and results of using dropout on the fully connected layer, first the CIFAR-10 dataset`s features are extracted using the convolutional layers. We will adopt the same network architectures to see if we can obtain similar results.

Hyperparameters can be obtained from [www.cs.toronto.edu](http://www.cs.toronto.edu/~nitish/dropout/). Given this knowledge, we can now straightforwardly build the architectures of the networks required for the reproduction and train them using dropout, implemented using both PyTorch and Keras. We believe the supplied codes are straightforward given previous and coming discussions and do not require further explanation besides the code itselves.


<a name="name_of_target"> </a>
# The Bernoulli and Gaussian Dropout Class 

Figures and explanations originate from [Srivastava, Hinton, Krizhevsky, Sutskever and Salakhutdinov (2014)](http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf). 

## Standard Activation Layer

Consider neural network with $L$ hidden layers. Let $l \in \{1, \dots, L\}$ index the hidden layers of the network. Let $\textbf{z}^{(l)}$ denote the vector of inputs into layer $l$, $\textbf{y}^{(l)}$ denote the vector of outputs from layer $l$ ($\textbf{y}^{(0)} = \textbf{x}$ is the input). $W^{(l)}$ and $\textbf{b}^{(l)}$ are the weigts and biases at layer $l$. The feed-forward operation of a standard neural network can be described as (for $l \in \{0, \dots, L-1\}$ and any hidden unit $i$)

\begin{aligned} 
z_i^{(l+1)} &= \textbf{w}_i^{(l+1)}\textbf{y}^l+b_i^{(l+1)} \\
y_i^{(l+1)} &= f(z_i^{(l+1)})
\end{aligned}

where $f$ is any activation function, for example 

$$ f(x) = \frac{1}{1+ \exp (-x))} $$

## (Standard) Bernoulli Dropout

For the Bernoulli Dropout adaptation we introduce a random variable $r_i^{(l)} \sim Bernoulli(p)$. The feed-forward then becomes

\begin{aligned}
\widetilde{\mathbf{y}}^{(l)} &=\mathbf{r}^{(l)} * \mathbf{y}^{(l)} \\
z_{i}^{(l+1)} &=\mathbf{w}_{i}^{(l+1)} \widetilde{\mathbf{y}}^{l}+b_{i}^{(l+1)} \\
y_{i}^{(l+1)} &=f(z_{i}^{(l+1)}) 
\end{aligned}

during **training**. 


<a title="By Srivastava et al. (2014)" href="https://leimao.github.io/images/blog/2019-06-04-Dropout-Explained/dropout.png"><img width="768" alt="Bernoulli Dropout" src="https://leimao.github.io/images/blog/2019-06-04-Dropout-Explained/dropout.png"/></a>


At **test time** we use a single neural net without dropout. The weights of this network are scaled down versions of the training weights. If a unit is retained with probability $p$ during training, the outgoing weights of that unit are multiplied by $p$ at test time $W_{test}^{(l)} = pW^{(l)}$. This ensure that the *expected* output (under the distribution used to drop units at training time) is the same as the actual output at test time. The feed-forwared step then becomes

\begin{aligned} 
z_i^{(l+1)} &= p\textbf{w}_i^{(l+1)}\textbf{y}^l+b_i^{(l+1)} \\
y_i^{(l+1)} &= f(z_i^{(l+1)})
\end{aligned}



## Inverted Bernoulli Dropout

Another way to achieve the same effect is to scale up the retained activations by multiplying by $\frac{1}{p}$ at **training time** and not modifying the weights at **test time**.

These methods are equivalent with appropriate scaling of the learning rate and weight initializations at each layer.

Therefore, dropout can be seen as a random variable $r_b$ that takes value $\frac{1}{p}$ with probability $p$ and 0 otherwise.

\begin{aligned} 
r_b &\sim \frac{1}{p} * Bernoulli(p) \\
\mathbb{E}[r_b] &= 1 \\
Var[r_b] &= \frac{1 - p}{p}
\end{aligned}

## (Inverted) Gaussian Dropout

We can adapt the inverted Bernoulli Dropout for the use of a Gaussian Dropout function. We create a Gaussian distributed random variable with zero mean and standard deviation equal to the activation unit. That is, each hidden activation $h_i$ is perturbed to $h_ir_g$ where $r_g \sim \mathcal{N}(1,1)$. We can generalize this to $r_g \sim \mathcal{N}(1, \sigma^2)$ where $\sigma$ is another hyperparameter to tune, just like $p$ in the Bernoulli dropout. If we set $\sigma^2 = \frac{1-p}{p}$ we obtain

\begin{aligned} 
r_g &\sim \mathcal{N}(1, \sigma^2) \\
\mathbb{E}[r_g] &= 1 \\
Var[r_g] &= \sigma^2 = \frac{1-p}{p}
\end{aligned}

Both the Bernoulli and Gaussian random variables have the same mean and variance. From the first and second order moments we know that $r_g$ has more entropy in comparison to $r_b$.

# Reproducibility Observations and Conclusions
The reproduction of results concerning the MNIST and CIFAR-10 datasets has been performed in both Pytorch as Keras. We believe the supplied codes are straightforward given previous discussions and do not require further explanation besides the code itselves. For both implementations we could reproduce the result for MNIST; meaning that the network we trained produced classification errors withing the bounds reported in table 10 at the start of this blogpost. For CIFAR-10 we unfortunately were unable to do so. 

The training procedure as used by (Srivastava et al.) for the MNIST dataset was to train "for a very long time". In the context of showing the effects of dropout this is not an unreasonable thought: When training longer, more of the $2^n$ effective network variations gets exploited. Although not stated explicitly in the paper, we strongly suspect the same was done for the CIFAR-10 dataset, in the same train of thought. This assumption is further strengthened by the fact that we experienced to lack the computational resources. The authors very likely had access to more computing power than we are able to compete with given our single GPU commodity system. As such, training time for the larger CIFAR-10 dataset became longer than we had expected when we started this reproduction, rendering experimentation within the bounds of the paper harder and especially more timeconsuming. The same problem does not occur with MNIST as it is a smaller and, as generally regarded, easier dataset. If one wants to attempt this reproduction, our advise and conclusion is that more computing power is required to practically do so.

Besides a lack of computing power, there might be other reasons the reproduction for CIFAR-10 was not a succes. Besides the hyperparameters [www.cs.toronto.edu](http://www.cs.toronto.edu/~nitish/dropout/) provides a fully customized implementation as well, not build upon libraries as TensorFlow, PyTorch or Keras (we tried to reproduce using the latter two). There could be differences in implementation that we are reasonably not aware of. Likewise, given our current (practical) knowledge, we might overlook a crucial details that would otherwise boost performance to be within the required bounds. 