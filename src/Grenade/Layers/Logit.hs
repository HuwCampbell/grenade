{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-|
Module      : Grenade.Layers.Logit
Description : Sigmoid nonlinear layer
Copyright   : (c) Huw Campbell, 2016-2017
License     : BSD2
Stability   : experimental
-}
module Grenade.Layers.Logit (
    Logit (..)
  ) where


import           Data.Serialize
import           Data.Singletons

import           Grenade.Core

-- | A Logit layer.
--
--   A layer which can act between any shape of the same dimension, perfoming an sigmoid function.
--   This layer should be used as the output layer of a network for logistic regression (classification)
--   problems.
data Logit = Logit
  deriving Show

instance UpdateLayer Logit where
  type Gradient Logit = ()
  runUpdate _ _ _ = Logit
  createRandom = return Logit

instance (a ~ b, SingI a) => Layer Logit a b where
  -- Wengert tape optimisation:
  --
  -- Derivative of the sigmoid function is
  --    d σ(x) / dx  = σ(x) • (1 - σ(x))
  -- but we have already calculated σ(x) in
  -- the forward pass, so just store that
  -- and use it in the backwards pass.
  type Tape Logit a b = S a
  runForwards _ a =
    let l = sigmoid a
    in  (l, l)
  runBackwards _ l g =
    let sigmoid' = l * (1 - l)
    in  ((), sigmoid' * g)

instance Serialize Logit where
  put _ = return ()
  get = return Logit

sigmoid :: Floating a => a -> a
sigmoid x = 1 / (1 + exp (-x))
