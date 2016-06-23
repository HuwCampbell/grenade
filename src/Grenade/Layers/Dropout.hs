{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE StandaloneDeriving    #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances     #-}

module Grenade.Layers.Dropout (
    Dropout (..)
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
data Dropout o = Dropout Double (R o)
  deriving Show

instance (MonadRandom m, KnownNat i) => Layer m (Dropout i) ('D1 i) ('D1 i) where
  runForwards _ _= error "todo"
  runBackards _ _ _ _ = error "todo"
