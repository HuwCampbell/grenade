{-# LANGUAGE DataKinds              #-}
{-# LANGUAGE GADTs                  #-}
{-# LANGUAGE TypeOperators          #-}
{-# LANGUAGE TypeFamilies           #-}
{-# LANGUAGE MultiParamTypeClasses  #-}
{-# LANGUAGE FlexibleContexts       #-}
{-# LANGUAGE RankNTypes             #-}
{-# LANGUAGE FlexibleInstances      #-}
{-# LANGUAGE FunctionalDependencies #-}
{-|
Module      : Grenade.Core.Layer.Accelerate
Description : Defines the Layer Classes required for the Accelerate backend
Copyright   : (c) Huw Campbell, 2016-2017
License     : BSD2
Stability   : experimental

This module defines what a Layer is in a Grenade
neural network.

There are two classes of interest: `UpdateLayer` and `Layer`.

`UpdateLayer` is required for all types which are used as a layer
in a network. Having no shape information, this class is agnotostic
to the input and output data of the layer.

An instance of `Layer` on the other hand is required for usage in
a neural network, but also specifies the shapes of data that the
network can transform. Multiple instance of `Layer` are permitted
for a single type, to transform different shapes. The `Reshape` layer
for example can act as a flattening layer, and its inverse, projecting
a 1D shape up to 2 or 3 dimensions.

Instances of `Layer` should be as strict as possible, and not emit
runtime errors.
-}
module Grenade.Core.Layer.Accelerate (
    Layer (..)
  , UpdateLayer (..)
  ) where

import           Data.List ( foldl' )
import           Data.Array.Accelerate

import qualified Grenade.Core.Shape as GS
import           Grenade.Core.LearningParameters.Accelerate
import           Grenade.Core.Matrix.Accelerate

-- | Class for updating a layer. All layers implement this, as it
--   describes how to create and update the layer.
--
class Accelerable l => UpdateLayer l where
  -- | The type for the gradient for this layer.
  --   Unit if there isn't a gradient to pass back.
  type Gradient l :: *

  -- | Update a layer with its gradient and learning parameters
  runUpdate       :: Acc LearningParameters -> (Accelerated l) -> Gradient l -> (Accelerated l)

  -- | Update a layer with many Gradients
  runUpdates      :: Acc LearningParameters -> (Accelerated l) -> [Gradient l] -> (Accelerated l)
  runUpdates rate = foldl' (runUpdate rate)

  {-# MINIMAL runUpdate #-}

-- | Class for a layer. All layers implement this, however, they don't
--   need to implement it for all shapes, only ones which are
--   appropriate.
--
class UpdateLayer l => Layer l (i :: GS.Shape) (o :: GS.Shape) where
  -- | The Wengert tape for this layer. Includes all that is required
  --   to generate the back propagated gradients efficiently. As a
  --   default, `S i` is fine.
  type Tape l i o :: *

  -- | Used in training and scoring. Take the input from the previous
  --   layer, and give the output from this layer.
  runForwards    :: (Accelerated l) -> Accelerated (GS.S i) -> (Tape l i o, (Accelerated (GS.S o)))

  -- | Back propagate a step. Takes the current layer, the input that
  --   the layer gave from the input and the back propagated derivatives
  --   from the layer above.
  --
  --   Returns the gradient layer and the derivatives to push back
  --   further.
  runBackwards   :: (Accelerated l) -> Tape l i o -> (Accelerated (GS.S o)) -> (Gradient l, Accelerated (GS.S i))
