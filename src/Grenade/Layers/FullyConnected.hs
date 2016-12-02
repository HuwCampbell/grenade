{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE RecordWildCards       #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances     #-}

module Grenade.Layers.FullyConnected (
    FullyConnected (..)
  , randomFullyConnected
  ) where

import           Control.Monad.Random hiding (fromList)

import           Data.Singletons.TypeLits

import           Numeric.LinearAlgebra.Static

import           Grenade.Core.Network
import           Grenade.Core.Shape

-- | A basic fully connected (or inner product) neural network layer.
data FullyConnected i o = FullyConnected
                        !(R o)   -- Bias neuron weights
                        !(L o i) -- Activation weights
                        !(L o i) -- Momentum

data FullyConnected' i o = FullyConnected'
                         !(R o)   -- Bias neuron gradient
                         !(L o i) -- Activation gradient

instance Show (FullyConnected i o) where
  show FullyConnected {} = "FullyConnected"

instance (Monad m, KnownNat i, KnownNat o) => UpdateLayer m (FullyConnected i o) where
  type Gradient (FullyConnected i o) = (FullyConnected' i o)

  runUpdate LearningParameters {..} (FullyConnected oldBias oldActivations oldMomentum) (FullyConnected' biasGradient activationGradient) = do
    let newBias        = oldBias - konst learningRate * biasGradient
        newMomentum    = konst learningMomentum * oldMomentum - konst learningRate * activationGradient
        regulariser    = konst (learningRegulariser * learningRate) * oldActivations
        newActivations = oldActivations + newMomentum - regulariser
    return $ FullyConnected newBias newActivations newMomentum

instance (Monad m, KnownNat i, KnownNat o) => Layer m (FullyConnected i o) ('D1 i) ('D1 o) where
  -- Do a matrix vector multiplication and return the result.
  runForwards (FullyConnected wB wN _) (S1D' v) = return $ S1D' (wB + wN #> v)

  -- Run a backpropogation step for a full connected layer.
  runBackards (FullyConnected _ wN _) (S1D' x) (S1D' dEdy) =
          let wB'  = dEdy
              mm'  = dEdy `outer` x
              -- calcluate derivatives for next step
              dWs  = tr wN #> dEdy
          in  return (FullyConnected' wB' mm', S1D' dWs)

randomFullyConnected :: (MonadRandom m, KnownNat i, KnownNat o)
                     => m (FullyConnected i o)
randomFullyConnected = do
    s1 :: Int <- getRandom
    s2 :: Int <- getRandom
    let wB = randomVector  s1 Uniform * 2 - 1
        wN = uniformSample s2 (-1) 1
        mm = konst 0
    return $ FullyConnected wB wN mm
