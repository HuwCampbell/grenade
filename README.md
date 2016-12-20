Grenade
=======

```
First shalt thou take out the Holy Pin, then shalt thou count to three, no more, no less.
Three shall be the number thou shalt count, and the number of the counting shall be three.
Four shalt thou not count, neither count thou two, excepting that thou then proceed to three.
Five is right out.
```

💣 Machine learning which might blow up in your face 💣

Grenade is a dependently typed, practical, and fast neural network library for
concise and precise specifications of complex networks in Haskell.

As an example, a network which can achieve ~1.5% error on MNIST can be
specified and initialised with random weights in a few lines of code with
```haskell
type MNIST = Network '[ Convolution 1 10 5 5 1 1, Pooling 2 2 2 2, Relu, Convolution 10 16 5 5 1 1, Pooling 2 2 2 2, FlattenLayer, Relu, FullyConnected 256 80, Logit, FullyConnected 80 10, Logit]
                     '[ 'D2 28 28, 'D3 24 24 10, 'D3 12 12 10, 'D3 12 12 10, 'D3 8 8 16, 'D3 4 4 16, 'D1 256, 'D1 256, 'D1 80, 'D1 80, 'D1 10, 'D1 10]

randomMnist :: MonadRandom m => m MNIST
randomMnist = randomNetwork
```

And that's it. Because the types are rich, there's no specific term level code
required; although it is of course possible and easy to construct one explicitly
oneself.

The network can be thought of as a heterogeneous list of layers, where its type
includes not only the layers of the network, but also the shapes of data that
are passed between the layers.

```haskell
data Network :: [*] -> [Shape] -> * where
    O     :: Layer x i o => !x -> Network '[x] '[i, o]
    (:~>) :: Layer x i h => !x -> !(Network xs (h ': hs)) -> Network (x ': xs) (i ': h ': hs)
```

The `Layer x i o` constraint ensures that the layer `x` can sensibly perform a
transformation between the input and output shapes `i` and `o`.

In the above example, the input layer can be seen to be a two dimensional (`D2`),
image with 28 by 28 pixels. When the first *Convolution* layer runs, it outputs
a three dimensional (`D3`) 24x24x10 image. The last item in the list is one
dimensional (`D1`) with 10 values, representing the categories of the MNIST
data.

Usage
-----

To perform back propagation, one can call the eponymous function
```haskell
backPropagate :: forall input target shapes layers. (Head shapes ~ input, Last shapes ~ target)
              => Network layers shapes -> S input -> S target -> Gradients layers
```
which takes a network, appropriate input and target data, and returns the
back propagated gradients for the network. The shapes of the gradients are
appropriate for each layer, and may be trivial for layers like `Rulu` which
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

Build Instructions
------------------
Grenade currently only builds with the [mafia](https://github.com/ambiata/mafia)
script that is located in the repository. You will also need the `lapack` and
`blas` libraries and development tools. Once you have all that, Grenade can be
build using:

```
./mafia build
```

and the tests run using:

```
./mafia test
```

Grenade is currently known to build with ghc 7.10 and 8.0.


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
