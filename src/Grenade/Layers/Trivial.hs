{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances     #-}
module Grenade.Layers.Trivial (
    Trivial (..)
  ) where

import           Data.Serialize

import           Grenade.Core

-- | A trivial layer.
data Trivial = Trivial
  deriving Show

instance Serialize Trivial where
  put _ = return ()
  get = return Trivial

instance UpdateLayer Trivial where
  type Gradient Trivial = ()
  runUpdate _ _ _ = Trivial
  createRandom = return Trivial

instance (a ~ b) => Layer Trivial a b where
  type Tape Trivial a b = ()
  runForwards _ a = ((), a)
  runBackwards _ _ y = ((), y)
