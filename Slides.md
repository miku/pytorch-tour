# Deep Learning with pytorch

## A short tour @lpyug 2017-02-14 19:00.

### Martin Czygan | github.com/miku/pytorch-tour

<!-- TODO: rename to dltour -->

----

## What is Deep Learning?

> Deep learning is a branch of machine learning based on a set of algorithms that attempt to model high-level abstractions in data by using multiple processing layers, with complex structures or otherwise, composed of multiple non-linear transformations.

----

## What is Deep Learning?

* maybe anything with more than one hidden layer
* computationally expensive, high capacity learning machines

----

## Overhyped?

> I'd say ML is both overhyped and underrated. People overestimate the intelligence and generalization power of ML systems (ML as a magic wand), but underestimate how much can be achieved with relatively crude systems, when applied systematically (ML as the steam power of our era). -- [950604](https://twitter.com/fchollet/status/950604227620950017)

> Deep Learning is under hyped compared to IoT, Big Data, Data Science, ChatBots and Robot Process Automation -- [950691](https://twitter.com/IntuitMachine/status/950691764553252865)

----

## In the news and elsewhere

* AlphaGo (March 2016, Deep Learning and the Game of Go)
* ImageNet classification (2014, VGG16, VGG19)
* Real-Time object detection (2013, darknet)
* Image Captioning
* Neural Style Transfer
* WaveNet
* Speech recognition (2017, [DeepSpeech](https://github.com/mozilla/DeepSpeech))
* Translation (2016, OpenNMT)
* Word Embeddings (2013, word2vec)

----

## And much more

* Image and scene generation
* Image segmentation
* Lip reading
* Text generation
* Time series forecast

----

## A tiny bit of money

In Nov 2007, Geoffrey Hinton gave a tech talk at Google, called *The Next Generation of Neural Networks*. He seems like a [slightly desperate](https://www.youtube.com/watch?v=AyzOUbkUf3M&feature=youtu.be&t=51m3s).

> We only trained this network once one one data set. If we could get a tiny bit of money from someone we could make this whole thing work much better.

Ten years later Hinton [introduces](https://www.utoronto.ca/news/introducing-vector-institute-ai-research) the Vector Institute at University of Toronto.

----

## Why Now?

* In short: data, cuda, relu.
* Or: availability of data, GPUS, algorithmic advances.

----

## Parts and Ingredients

Build a **computational graph**, utilize **automatic differentiation**, to adjust the **parameters** of your model according to a given **loss function**, that captures the **distance** between the computer and the expected output, when given enough **training data**.