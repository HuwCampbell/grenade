module Grenade.Recurrent (
  -- | This is an empty module which simply re-exports public definitions
  --   for recurrent networks in Grenade.

  -- * Exported modules
  --
  -- | The core types and runners for Recurrent Networks.
    module Grenade.Recurrent.Core

  -- | The recurrent neural network layer zoo
  , module Grenade.Recurrent.Layers

    -- * Overview of recurrent Networks
    -- $recurrent

  ) where

import           Grenade.Recurrent.Core
import           Grenade.Recurrent.Layers

{- $recurrent
There are two ways in which deep learning libraries choose to represent
recurrent Neural Networks, as an unrolled graph, or at a first class
level. Grenade chooses the latter representation, and provides a network
type which is specifically suited for recurrent neural networks.

Currently grenade supports two layers, a basic recurrent layer, and an
LSTM layer.
-}
