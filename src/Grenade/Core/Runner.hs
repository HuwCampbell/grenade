{-# LANGUAGE DataKinds        #-}
{-# LANGUAGE FlexibleContexts #-}
{-|
Module      : Grenade.Core.Runner
Description : Functions to perform training and backpropagation
Copyright   : (c) Huw Campbell, 2016-2017
License     : BSD2
Stability   : experimental
-}
module Grenade.Core.Runner (
    train
  , backPropagate
  , runNet
  ) where

import           Data.Singletons.Prelude

import           Grenade.Core.Network
import           Grenade.Core.Optimizer
import           Grenade.Core.Shape

-- | Perform reverse automatic differentiation on the network
--   for the current input and expected output.
--
--   /Note:/ The loss function pushed backwards is appropriate
--   for both regression and classification as a squared loss
--   or log-loss respectively.
--
--   For other loss functions, use runNetwork and runGradient
--   with the back propagated gradient of your loss.
--
backPropagate :: (SingI (Last shapes))
              => Network layers shapes
              -> S (Head shapes)
              -> S (Last shapes)
              -> Gradients layers
backPropagate network input target =
    let (tapes, output) = runNetwork network input
        (grads, _)      = runGradient network tapes (output - target)
    in  grads


-- | Update a network with new weights after training with an instance.
train :: (SingI (Last shapes))
      => Optimizer opt
      -> Network layers shapes
      -> S (Head shapes)
      -> S (Last shapes)
      -> Network layers shapes
train optimizer network input output =
    let grads = backPropagate network input output
    in  applyUpdate optimizer network grads


-- | Run the network with input and return the given output.
runNet :: Network layers shapes -> S (Head shapes) -> S (Last shapes)
runNet net = snd . runNetwork net
