Deep Graph with PyTorch
=======================

* https://www.youtube.com/watch?v=ZUHhNuw9Tlc

Neural coming-of-age story. How do researchers come up with these architectures.

> Evolution of a neuron into a hundreds.

* 1957 Rosenblatt
* Minsky, Papert, Perceptron, 1969 Book, Critique, can only distinguish linear
  pattern
* 2015, ResNet-152, 152 layer deep network
* 1991, serious problems
* 2012, first frameworks
* 2015, a smart idea - residual networks

----

* NN, DL: function approximators, Universal approximation theorem
* http://neuralnetworksanddeeplearning.com/chap4.html

> One of the most striking facts about neural networks is that they can compute
> any function at all. That is, suppose someone hands you some complicated,
> wiggly function, f(x) ... No matter what the function, there is guaranteed to
> be a neural network so that for every possible input, x, the value f(x) (or
> some close approximation) is output from the network, e.g.

> This result tells us that neural networks have a kind of universality. No
> matter what function we want to compute, we know that there is a neural
> network which can do the job.

> Universality theorems are a commonplace in computer science, so much so that
> we sometimes forget how astonishing they are. But it's worth reminding
> ourselves: the ability to compute an arbitrary function is truly remarkable.
> Almost any process you can imagine can be thought of as function computation.

> Summing up, a more precise statement of the universality theorem is that
> neural networks with a single hidden layer can be used to approximate any
> continuous function to any desired precision.

----

* Classification (e.g. input image, output label)
* Generative (e.g. blurry input image, sharper output similar image)

Examples:

* GalaxyGAN, space.ml/proj/GalaxyGAN
* CellProfiler Analyst, Classfication of human cells, cellprofiler.org, Dao et. al. 2015
* Reinforcement learning, street image as input - every action has consequences
  for the future, always a little bit off, might accumulate errors, instead: try
  to learn a policy by looking at the complete path,
  https://blogs.nvidia.com/blog/2016/05/06/self-driving-cars-3/

----

Back to numpy. Pure numpy perspective. No AD.

