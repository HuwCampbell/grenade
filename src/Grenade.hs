module Grenade (
  -- | This is an empty module which simply re-exports public definitions
  --   for machine learning with Grenade.

  -- * Exported modules
  --
  -- | The core types and runners for Grenade.
    module Grenade.Core

  -- | The neural network layer zoo
  , module Grenade.Layers


    -- * Overview of the library
    -- $library

    -- * Example usage
    -- $example

  ) where

import           Grenade.Core
import           Grenade.Layers

{- $library
Grenade is a purely functional deep learning library.

It provides an expressive type level API for the construction
of complex neural network architectures. Backing this API is and
implementation written using BLAS and LAPACK, mostly provided by
the hmatrix library.

-}

{- $example
A few examples are provided at https://github.com/HuwCampbell/grenade
under the examples folder.

The starting place is to write your neural network type and a
function to create a random layer of that type. The following
is a simple example which runs a logistic regression.

> type MyNet = Network '[ FullyConnected 10 1, Logit ] '[ 'D1 10, 'D1 1, 'D1 1 ]
>
> randomMyNet :: MonadRandom MyNet
> randomMyNet = randomNetwork

The function `randomMyNet` witnesses the `CreatableNetwork`
constraint of the neural network, and in doing so, ensures the network
can be built, and hence, that the architecture is sound.
-}


