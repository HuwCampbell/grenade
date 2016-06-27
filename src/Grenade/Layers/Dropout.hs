{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE StandaloneDeriving    #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE LambdaCase            #-}

module Grenade.Layers.Dropout (
    Dropout (..)
  ) where

import           Control.Monad.Random hiding (fromList)
import           Control.Monad.State

import           GHC.TypeLits
import           Grenade.Core.Shape
import           Grenade.Core.Network
import           Grenade.Core.Phase

import           Numeric.LinearAlgebra.Static


-- Dropout layer help to reduce overfitting.
-- Idea here is that the vector is a shape of 1s and 0s, which we multiply the input by.
-- After backpropogation, we return a new matrix/vector, with different bits dropped out.
-- Double is the proportion to drop in each training iteration (like 1% or 5% would be
-- reasonable).
data Dropout o = Dropout Double (R o)
  deriving Show

randomDropout :: (MonadRandom m, KnownNat i)
              => Double -> m (Dropout i)
randomDropout rate = do
    seed  <- getRandom
    let wN = randomVector seed Uniform
        xs = dvmap (\a -> if a <= rate then 0 else 1) wN
    return $ Dropout rate xs

instance (MonadRandom m, MonadState Phase m, KnownNat i) => Layer m (Dropout i) ('D1 i) ('D1 i) where
  runForwards (Dropout rate drops) (S1D' x) = isTrainingPhase >>= \case
    True  -> return . S1D' $ x * drops
    False -> return . S1D' $ dvmap (* (1 - rate)) x
  runBackards _ oldDropout@(Dropout rate drops) _ (S1D' x) = isTrainingPhase >>= \case
    True -> do
      newDropout <- randomDropout rate
      return (newDropout,  S1D' $ x * drops)
    False -> return (oldDropout,  S1D' $  dvmap (* (1 - rate)) x)
