{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE FlexibleInstances     #-}
module Grenade.Recurrent.Core.Layer (
    RecurrentLayer (..)
  , RecurrentUpdateLayer (..)
  ) where

import           Data.Singletons ( SingI )

import           Grenade.Core

-- | Class for a recurrent layer.
--   It's quite similar to a normal layer but for the input and output
--   of an extra recurrent data shape.
class UpdateLayer x => RecurrentUpdateLayer x where
  -- | Shape of data that is passed between each subsequent run of the layer
  type RecurrentShape x   :: Shape

class (RecurrentUpdateLayer x, SingI (RecurrentShape x)) => RecurrentLayer x (i :: Shape) (o :: Shape) where
  -- | Wengert Tape
  type RecTape x i o :: *
  -- | Used in training and scoring. Take the input from the previous
  --   layer, and give the output from this layer.
  runRecurrentForwards    :: x -> S (RecurrentShape x) -> S i -> (RecTape x i o, S (RecurrentShape x), S o)
  -- | Back propagate a step. Takes the current layer, the input that the
  --   layer gave from the input and the back propagated derivatives from
  --   the layer above.
  --   Returns the gradient layer and the derivatives to push back further.
  runRecurrentBackwards   :: x -> RecTape x i o -> S (RecurrentShape x) -> S o -> (Gradient x, S (RecurrentShape x), S i)
