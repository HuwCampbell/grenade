{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE TypeFamilies          #-}

module Grenade.Core.Runner (
    train
  , runNet
  ) where

import           Data.Singletons.Prelude
import           Grenade.Core.Network
import           Grenade.Core.Shape

-- | Update a network with new weights after training with an instance.
train :: forall m i o hs. (Monad m, Head hs ~ i, Last hs ~ o, KnownShape i, KnownShape o)
      => LearningParameters   -- ^ learning rate
      -> S' i                 -- ^ input vector
      -> S' o                 -- ^ target vector
      -> Network m hs         -- ^ network to train
      -> m (Network m hs)
train rate x0 target = fmap fst . go x0
  where
    go  :: forall m' j js. (Monad m', Head js ~ j, Last js ~ o, KnownShape j)
        => S' j                -- ^ input vector
        -> Network m' js       -- ^ network to train
        -> m' (Network m' js, S' j)
    -- handle input from the beginning, feeding upwards.
    go !x (layer :~> n)
        = do y             <- runForwards layer x
              -- run the rest of the network, and get the layer from above.
             (n', dWs')    <- go y n
              -- calculate the gradient for this layer to pass down,
             (layer', dWs) <- runBackards layer x dWs'

             -- Update this layer using the gradient
             newLayer      <- runUpdate rate layer layer'

             return (newLayer :~> n', dWs)

    -- handle the output layer, bouncing the derivatives back down.
    go !x (O layer)
        = do y                 <- runForwards layer x
              -- the gradient (how much y affects the error)
             (layer', dWs)     <- runBackards layer x (y - target)
             newLayer          <- runUpdate rate layer layer'

             return (O newLayer, dWs)

-- | Just forwards propagation with no training.
runNet :: forall m hs. (Monad m)
       => Network m hs
       -> S' (Head hs)         -- ^ input vector
       -> m (S' (Last hs))     -- ^ target vector
runNet (layer :~> n)  !x = do y <- runForwards layer x
                              runNet n y
runNet (O layer)      !x = runForwards layer x
