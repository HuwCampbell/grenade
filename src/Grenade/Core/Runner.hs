{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE TypeFamilies          #-}
{-|
Module      : Grenade.Core.Shape
Description : Core definition of the Shapes of data we understand
Copyright   : (c) Huw Campbell, 2016-2017
License     : BSD2
Stability   : experimental

This module defines simple back propagation and training functions
for a network.
-}
module Grenade.Core.Runner (
    train
  , backPropagate
  , runNet
  , applyUpdate
  ) where

import           Data.Singletons.Prelude
import           Grenade.Core.Network
import           Grenade.Core.Shape

-- | Perform reverse automatic differentiation on the network
--   for the current input and expected output.
--
--   /Note:/ The loss function pushed backwards is appropriate
--   for both regression and classification as a squared loss
--   or log-loss respectively. Other loss functions are not yet
--   implemented.
backPropagate :: forall shapes layers.
                 Network layers shapes -> S (Head shapes) -> S (Last shapes) -> Gradients layers
backPropagate network input target =
    fst $ go input network
  where
    go  :: forall js sublayers. (Last js ~ Last shapes)
        => S (Head js)          -- ^ input vector
        -> Network sublayers js -- ^ network to train
        -> (Gradients sublayers, S (Head js))
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

-- | Apply one step of stochastic gradient decent across the network.
applyUpdate :: LearningParameters -> Network ls ss -> Gradients ls -> Network ls ss
applyUpdate rate (O layer) (OG gradient)
  = O (runUpdate rate layer gradient)
applyUpdate rate (layer :~> rest) (gradient :/> grest)
  = runUpdate rate layer gradient :~> applyUpdate rate rest grest
applyUpdate _ _ _
  = error "Impossible for the gradients of a network to have a different length to the network"

-- | Update a network with new weights after training with an instance.
train :: LearningParameters -> Network layers shapes -> S (Head shapes) -> S (Last shapes) -> Network layers shapes
train rate network input output =
    let grads = backPropagate network input output
    in  applyUpdate rate network grads

-- | Run the network with input and return the given output.
runNet :: Network layers shapes -> S (Head shapes) -> S (Last shapes)
runNet (layer :~> n)  !x = let y = runForwards layer x in runNet n y
runNet (O layer)      !x = runForwards layer x
