Grenade
=======

[![Build Status](https://api.travis-ci.com/schnecki/grenade.svg?branch=master)](https://travis-ci.org/schnecki/grenade)
<!-- [![Hackage page (downloads and API reference)][hackage-png]][hackage] -->
<!-- [![Hackage-Deps][hackage-deps-png]][hackage-deps] -->

This is a fork of the original Grenade library found at https://github.com/HuwCampbell/grenade,
but includes additional features:

 0. **OpenMP Parallelisation**: You need to have OpenMP installed (e.g. `sudo pacman -S openmp`) to
    compile the code, as the c-code uses OpenMP for-loop parallelisation.

 1. **Optimizer Support**. The code has been restructured to be able to easily implement more
    optimizers than just SGD with momentum and regularization. Currently we support *Adam* for
    feedforward neural networks also!

 2. **Weight Initialization**. Initializing the weights in different ways. Currently implemented:
    Uniform, HeEtAl, Xavier. The default is Uniform! See chapter 6 of this [seminar
    report](docs/seminar_report_ANN_analysis_2018.pdf "Seminar Report ANN Analysis") for a small
    evaluation of the implemented weight initialization methods.

 3. **Data Type Representation**: You can easily switch from `Double` to `Float` vectors and
    matrices. Just provide the corresponding flag (`use-float`) when compiling (to all packages
    that):

        stack clean && stack build --flag=grenade-examples:use-float --flag=grenade:use-float && stack bench

    Ensure you clean before changing the flags, as otherwise you might in the best case get a
    compile error and in the worst case a Segmentation Fault!

    Clearly `Float`s are less precise but more efficient, both in terms of *time and memory*. In
    case of small ANNs `Float` should be sufficient, as long as you keep the values of the weights
    small (which you should always do). This feature uses an [adapted
    version](http://github.com/schnecki/hmatrix-float "github repository") of
    [hmatrix](https://hackage.haskell.org/package/hmatrix-0.20.0.0 "stackage") which was especially
    adapted for this project.

 4. **Runtime Networks**. Dynamically specifying and build networks at runtime. This is not only a
    required tool when storing the network architecture to the disk, like in a DB, and reloading it,
    but it could also be a starting point for developing algorithms that adapt the network to find
    the best architecture for the underlying problem. You can do that with this feature without
    knowing its structure by deserializing the network specification and then feed the deserialized
    network weights into the net.

    However, currently this works only for feedforward networks composed of fully-connected, dropout,
    deconvolution and convolution layers plus all activation functions. Example (also see
    `feedforward-netinit` in example folder):
    ```haskell
       let spec :: SpecNet
           spec = specFullyConnected 40 30 |=> specRelu1D 30 |=> specFullyConnected 30 20 |=> specNil1D 20
       SpecConcreteNetwork1D1D (net0 :: Network layers shapes) <- networkFromSpecificationWith HeEtAl spec
    ```

    However, Beware! It is important to get the specification right, as otherwise the program will halt
    abruptly. So at best do not use it manually, but write functions for creating specifications!

    Or probably better, use the simple interface:

    ```haskell
       buildNetViaInterface :: IO SpecConcreteNetwork
       buildNetViaInterface =
         buildModel $
         inputLayer1D 2 >>
         fullyConnected 10 >> dropout 0.89 >> relu >>
         fullyConnected 4 >> relu >>
         networkLayer (
           inputLayer1D 4 >> fullyConnected 10 >> relu >> fullyConnected 4 >> sinusoid) >>
         fullyConnected 1 >> tanhLayer
    ```
 5. **Gradient Clipping**. You can clip gradients using the function `clipByGlobalNorm`.

 6. **More Activation Functions**. This branch supports `Dropout` (which is unimplemented in the
    original code), `LeakyRelu` and `Gelu` activation functions.


The following is mostly (except the installation procedure) the original description of grenade:

Description
===========

```
First shalt thou take out the Holy Pin, then shalt thou count to three, no more, no less.
Three shall be the number thou shalt count, and the number of the counting shall be three.
Four shalt thou not count, neither count thou two, excepting that thou then proceed to three.
Five is right out.
```

💣 Machine learning which might blow up in your face 💣

Grenade is a composable, dependently typed, practical, and fast recurrent neural network library
for concise and precise specifications of complex networks in Haskell.

As an example, a network which can achieve ~1.5% error on MNIST can be
specified and initialised with random weights in a few lines of code with
```haskell
type MNIST
  = Network
    '[ Convolution 1 10 5 5 1 1, Pooling 2 2 2 2, Relu
     , Convolution 10 16 5 5 1 1, Pooling 2 2 2 2, Reshape, Relu
     , FullyConnected 256 80, Logit, FullyConnected 80 10, Logit]
    '[ 'D2 28 28, 'D3 24 24 10, 'D3 12 12 10, 'D3 12 12 10
     , 'D3 8 8 16, 'D3 4 4 16, 'D1 256, 'D1 256
     , 'D1 80, 'D1 80, 'D1 10, 'D1 10]

randomMnist :: MonadRandom m => m MNIST
randomMnist = randomNetwork
```

And that's it. Because the types are so rich, there's no specific term level code
required to construct this network; although it is of course possible and
easy to construct and deconstruct the networks and layers explicitly oneself.

If recurrent neural networks are more your style, you can try defining something
["unreasonably effective"](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
with
```haskell
type Shakespeare
  = RecurrentNetwork
    '[ R (LSTM 40 80), R (LSTM 80 40), F (FullyConnected 40 40), F Logit]
    '[ 'D1 40, 'D1 80, 'D1 40, 'D1 40, 'D1 40 ]
```

Design
------

Networks in Grenade can be thought of as a heterogeneous lists of layers, where
their type includes not only the layers of the network, but also the shapes of
data that are passed between the layers.

The definition of a network is surprisingly simple:
```haskell
data Network :: [*] -> [Shape] -> * where
    NNil  :: SingI i
          => Network '[] '[i]

    (:~>) :: (SingI i, SingI h, Layer x i h)
          => !x
          -> !(Network xs (h ': hs))
          -> Network (x ': xs) (i ': h ': hs)
```

The `Layer x i o` constraint ensures that the layer `x` can sensibly perform a
transformation between the input and output shapes `i` and `o`.

The lifted data kind `Shape` defines our 1, 2, and 3 dimension types, used to
declare what shape of data is passed between the layers.

In the MNIST example above, the input layer can be seen to be a two dimensional
(`D2`), image with 28 by 28 pixels. When the first *Convolution* layer runs, it
outputs a three dimensional (`D3`) 24x24x10 image. The last item in the list is
one dimensional (`D1`) with 10 values, representing the categories of the MNIST
data.

Usage
-----

To perform back propagation, one can call the eponymous function
```haskell
backPropagate :: forall shapes layers.
                 Network layers shapes -> S (Head shapes) -> S (Last shapes) -> Gradients layers
```
which takes a network, appropriate input and target data, and returns the
back propagated gradients for the network. The shapes of the gradients are
appropriate for each layer, and may be trivial for layers like `Relu` which
have no learnable parameters.

The gradients however can always be applied, yielding a new (hopefully better)
layer with
```haskell
applyUpdate :: LearningParameters -> Network ls ss -> Gradients ls -> Network ls ss
```

Layers in Grenade are represented as Haskell classes, so creating one's own is
easy in downstream code. If the shapes of a network are not specified correctly
and a layer can not sensibly perform the operation between two shapes, then
it will result in a compile time error.

Composition
-----------

Networks and Layers in Grenade are easily composed at the type level. As a `Network`
is an instance of `Layer`, one can use a trained Network as a small component in a
larger network easily. Furthermore, we provide 2 layers which are designed to run
layers in parallel and merge their output (either by concatenating them across one
dimension or summing by pointwise adding their activations). This allows one to
write any Network which can be expressed as a
[series parallel graph](https://en.wikipedia.org/wiki/Series-parallel_graph).

A residual network layer specification for instance could be written as
```haskell
type Residual net = Merge Trivial net
```
If the type `net` is an instance of `Layer`, then `Residual net` will be too. It will
run the network, while retaining its input by passing it through the `Trivial` layer,
and merge the original image with the output.

See the [MNIST](https://github.com/HuwCampbell/grenade/blob/master/examples/main/mnist.hs)
example, which has been overengineered to contain both residual style learning as well
as inception style convolutions.

Generative Adversarial Networks
-------------------------------

As Grenade is purely functional, one can compose its training functions in flexible
ways. [GAN-MNIST](https://github.com/HuwCampbell/grenade/blob/master/examples/main/gan-mnist.hs)
example displays an interesting, type safe way of writing a generative adversarial
training function in 10 lines of code.

Layer Zoo
---------

Grenade layers are normal haskell data types which are an instance of `Layer`, so
it's easy to build one's own downstream code. We do however provide a decent set
of layers, including convolution, deconvolution, pooling, pad, crop, logit, relu,
elu, tanh, and fully connected.

Build Instructions
------------------
This version of Grenade is most easily built with the. You will also need the `lapack` and
`blas` libraries and development tools. Once you have all that, Grenade can be
build using:

```
stack build
```

and the tests run using:

```
stack test
```

This version of Grenade builds with GHC 8.8.

Thanks
------
Writing a library like this has been on my mind for a while now, but a big shout
out must go to [Justin Le](https://github.com/mstksg), whose
[dependently typed fully connected network](https://blog.jle.im/entry/practical-dependent-types-in-haskell-1.html)
inspired me to get cracking, gave many ideas for the type level tools I
needed, and was a great starting point for writing this library.

Performance
-----------
Grenade is backed by hmatrix, BLAS, and LAPACK, with critical functions optimised
in C. Using the im2col trick popularised by Caffe, it should be sufficient for
many problems.

Being purely functional, it should also be easy to run batches in parallel, which
would be appropriate for larger networks, my current examples however are single
threaded.

Training 15 generations over Kaggle's 41000 sample MNIST training set on a single
core took around 12 minutes, achieving 1.5% error rate on a 1000 sample holdout set.

Contributing
------------
Contributions are welcome.

 [hackage]: http://hackage.haskell.org/package/grenade
 [hackage-png]: http://img.shields.io/hackage/v/grenade.svg
 [hackage-deps]: http://packdeps.haskellers.com/reverse/grenade
 [hackage-deps-png]: https://img.shields.io/hackage-deps/v/grenade.svg
