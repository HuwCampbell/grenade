{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances     #-}
module Grenade.Recurrent.Layers.Trivial (
    Trivial (..)
  ) where

import           Grenade.Core.Network

-- | A trivial layer.
data Trivial = Trivial
  deriving Show

instance UpdateLayer Trivial where
  type Gradient Trivial = ()
  runUpdate _ _ _ = Trivial
  createRandom = return Trivial

instance (a ~ b) => Layer Trivial a b where
  runForwards _ = id
  runBackwards _ _ y = ((), y)
