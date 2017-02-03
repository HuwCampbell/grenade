{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE FlexibleInstances     #-}
{-|
Module      : Grenade.Core.Network
Description : Core definition a simple neural etwork
Copyright   : (c) Huw Campbell, 2016-2017
License     : BSD2
Stability   : experimental

This module defines what a Layer is in a Grenade
neural network.
-}
module Grenade.Core.Layer (
    Layer (..)
  , UpdateLayer (..)
  ) where

import           Control.Monad.Random ( MonadRandom )

import           Data.List ( foldl' )

import           Grenade.Core.Shape
import           Grenade.Core.LearningParameters

-- | Class for updating a layer. All layers implement this, as it
--   describes how to create and update the layer.
--
class UpdateLayer x where
  -- | The type for the gradient for this layer.
  --   Unit if there isn't a gradient to pass back.
  type Gradient x :: *

  -- | Update a layer with its gradient and learning parameters
  runUpdate       :: LearningParameters -> x -> Gradient x -> x

  -- | Create a random layer, many layers will use pure
  createRandom    :: MonadRandom m => m x

  -- | Update a layer with many Gradients
  runUpdates      :: LearningParameters -> x -> [Gradient x] -> x
  runUpdates rate = foldl' (runUpdate rate)

  {-# MINIMAL runUpdate, createRandom #-}

-- | Class for a layer. All layers implement this, however, they don't
--   need to implement it for all shapes, only ones which are
--   appropriate.
--
class UpdateLayer x => Layer x (i :: Shape) (o :: Shape) where
  -- | The Wengert tape for this layer. Includes all that is required
  --   to generate the back propagated gradients efficiently. As a
  --   default, `S i` is fine.
  type Tape x i o :: *

  -- | Used in training and scoring. Take the input from the previous
  --   layer, and give the output from this layer.
  runForwards    :: x -> S i -> (Tape x i o, S o)

  -- | Back propagate a step. Takes the current layer, the input that
  --   the layer gave from the input and the back propagated derivatives
  --   from the layer above.
  --
  --   Returns the gradient layer and the derivatives to push back
  --   further.
  runBackwards   :: x -> Tape x i o -> S o -> (Gradient x, S i)
