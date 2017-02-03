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
import           Grenade.Core

-- Dropout layer help to reduce overfitting.
-- Idea here is that the vector is a shape of 1s and 0s, which we multiply the input by.
-- After backpropogation, we return a new matrix/vector, with different bits dropped out.
-- Double is the proportion to drop in each training iteration (like 1% or 5% would be
-- reasonable).
data Dropout = Dropout {
    dropoutRate :: Double
  , dropoutSeed :: Int
  } deriving Show

instance UpdateLayer Dropout where
  type Gradient Dropout = ()
  runUpdate _ x _ = x
  createRandom = randomDropout 0.95

randomDropout :: MonadRandom m
              => Double -> m Dropout
randomDropout rate = Dropout rate <$> getRandom

instance (KnownNat i) => Layer Dropout ('D1 i) ('D1 i) where
  type Tape Dropout ('D1 i) ('D1 i) = ()
  runForwards (Dropout _ _) (S1D x) = ((), S1D x)
  runBackwards (Dropout _ _) _ (S1D x) = ((),  S1D x)
