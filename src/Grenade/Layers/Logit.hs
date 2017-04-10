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
  type Tape Logit a b = S a
  runForwards _ a = (a, logistic a)
  runBackwards _ a g = ((), logistic' a * g)

instance Serialize Logit where
  put _ = return ()
  get = return Logit

logistic :: Floating a => a -> a
logistic x = 1 / (1 + exp (-x))

logistic' :: Floating a => a -> a
logistic' x = logix * (1 - logix)
  where
    logix = logistic x
