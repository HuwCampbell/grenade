{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TypeOperators         #-}
{-|
Module      : Grenade.Layers.Tanh
Description : Hyperbolic tangent nonlinear layer
Copyright   : (c) Huw Campbell, 2016-2017
License     : BSD2
Stability   : experimental
-}
module Grenade.Layers.Tanh (
    Tanh (..)
  ) where

import           Data.Serialize
import           Data.Singletons

import           Grenade.Core

-- | A Tanh layer.
--   A layer which can act between any shape of the same dimension, performing a tanh function.
data Tanh = Tanh
  deriving Show

instance UpdateLayer Tanh where
  type Gradient Tanh = ()
  runUpdate _ _ _ = Tanh
  createRandom = return Tanh

instance Serialize Tanh where
  put _ = return ()
  get = return Tanh

instance (a ~ b, SingI a) => Layer Tanh a b where
  type Tape Tanh a b = S a
  runForwards _ a = (a, tanh a)
  runBackwards _ a g = ((), tanh' a * g)

tanh' :: (Floating a) => a -> a
tanh' t = 1 - s ^ (2 :: Int)  where s = tanh t
