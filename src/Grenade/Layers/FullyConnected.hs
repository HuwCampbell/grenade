{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE StandaloneDeriving    #-}
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
                        !(L o i) -- Activation momentums

instance Show (FullyConnected i o) where
  show (FullyConnected _ _ _) = "FullyConnected"

instance (Monad m, KnownNat i, KnownNat o) => Layer m (FullyConnected i o) ('D1 i) ('D1 o) where
  -- Do a matrix vector multiplication and return the result.
  runForwards (FullyConnected wB wN _) (S1D' v) = return $ S1D' (wB + wN #> v)

  -- Run a backpropogation step for a full connected layer.
  runBackards rate (FullyConnected wB wN mm) (S1D' x) (S1D' dEdy) =
          let wB'  = wB - konst rate * dEdy
              mm'  = 0.9 * mm - konst rate * (dEdy `outer` x)
              wd'  = konst (0.0001 * rate) * wN
              wN'  = wN + mm' - wd'
              w'   = FullyConnected wB' wN' mm'
              -- calcluate derivatives for next step
              dWs  = tr wN #> dEdy
          in  return (w', S1D' dWs)

randomFullyConnected :: (MonadRandom m, KnownNat i, KnownNat o)
                     => m (FullyConnected i o)
randomFullyConnected = do
    s1 :: Int <- getRandom
    s2 :: Int <- getRandom
    let wB = randomVector  s1 Uniform * 2 - 1
        wN = uniformSample s2 (-1) 1
        mm = konst 0
    return $ FullyConnected wB wN mm
