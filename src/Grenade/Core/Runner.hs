{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE TypeFamilies          #-}

module Grenade.Core.Runner (
    train
  , backPropagate
  , runNet
  , applyUpdate
  ) where

import           Data.Singletons.Prelude
import           Grenade.Core.Network
import           Grenade.Core.Shape

-- | Drive and network and collect its back propogated gradients.
backPropagate :: forall input output shapes layers. (Head shapes ~ input, Last shapes ~ output)
              => Network layers shapes -> S' input -> S' output -> Gradients layers
backPropagate network input target =
    fst $ go input network
  where
    go  :: forall j js sublayers. (Head js ~ j, Last js ~ output)
        => S' j                 -- ^ input vector
        -> Network sublayers js -- ^ network to train
        -> (Gradients sublayers, S' j)
    -- handle input from the beginning, feeding upwards.
    go !x (layer :~> n)
        = let y             = runForwards layer x
              -- recursively run the rest of the network, and get the gradients from above.
              (n', dWs')    = go y n
              -- calculate the gradient for this layer to pass down,
              (layer', dWs) = runBackwards layer x dWs'

          in (layer' :/> n', dWs)

    -- handle the output layer, bouncing the derivatives back down.
    go !x (O layer)
        = let y                 = runForwards layer x
            -- the gradient (how much y affects the error)
              (layer', dWs)     = runBackwards layer x (y - target)

          in (OG layer', dWs)

-- | Update a network with new weights after training with an instance.
train :: forall input output shapes layers. (Head shapes ~ input, Last shapes ~ output)
      => LearningParameters            -- ^ learning rate
      -> Network layers shapes         -- ^ network to train
      -> S' input -> S' output         -- ^ target vector
      -> Network layers shapes
train rate network input output =
    let grads = backPropagate network input output
    in  applyUpdate rate network grads

applyUpdate :: LearningParameters -> Network ls ss -> Gradients ls -> Network ls ss
applyUpdate rate (O layer) (OG gradient)
  = O (runUpdate rate layer gradient)
applyUpdate rate (layer :~> rest) (gradient :/> grest)
  = runUpdate rate layer gradient :~> applyUpdate rate rest grest
applyUpdate _ _ _
  = error "Impossible for the gradients of a network to have a different length to the network"

-- | Just forwards propagation with no training.
runNet :: Network layers hs
       -> S' (Head hs)         -- ^ input vector
       -> S' (Last hs)         -- ^ target vector
runNet (layer :~> n)  !x = let y = runForwards layer x in runNet n y
runNet (O layer)      !x = runForwards layer x
