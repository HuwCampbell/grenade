{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
module Grenade.Layers.Dropout (
    Dropout (..)
  , randomDropout
  ) where

import           Control.Monad.Random hiding (fromList)

import           GHC.TypeLits
import           Grenade.Core.Shape
import           Grenade.Core.Network

import           Numeric.LinearAlgebra.Static


-- Dropout layer help to reduce overfitting.
-- Idea here is that the vector is a shape of 1s and 0s, which we multiply the input by.
-- After backpropogation, we return a new matrix/vector, with different bits dropped out.
-- Double is the proportion to drop in each training iteration (like 1% or 5% would be
-- reasonable).
data Dropout o =
  Dropout (R o)
  deriving Show

instance (KnownNat i) => UpdateLayer (Dropout i) where
  type Gradient (Dropout i) = ()
  runUpdate _ x _ = x
  createRandom = randomDropout 0.95

randomDropout :: (MonadRandom m, KnownNat i)
              => Double -> m (Dropout i)
randomDropout rate = do
    seed  <- getRandom
    let wN = randomVector seed Uniform
        xs = dvmap (\a -> if a <= rate then 0 else 1) wN
    return $ Dropout xs

instance (KnownNat i) => Layer (Dropout i) ('D1 i) ('D1 i) where
  type Tape (Dropout i) ('D1 i) ('D1 i) = ()
  runForwards (Dropout drops) (S1D x) = ((), S1D $ x * drops)
  runBackwards (Dropout drops) _ (S1D x) = ((),  S1D $ x * drops)
