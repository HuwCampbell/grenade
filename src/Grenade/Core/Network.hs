{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE KindSignatures        #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE PolyKinds             #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE FlexibleInstances     #-}

module Grenade.Core.Network (
    Layer (..)
  , Network (..)
  , UpdateLayer (..)
  , LearningParameters (..)
  ) where

import           Grenade.Core.Shape

data LearningParameters = LearningParameters {
    learningRate :: Double
  , learningMomentum :: Double
  , learningRegulariser :: Double
  } deriving (Eq, Show)

-- | Class for updating a layer. All layers implement this, and it is
--   shape independent.
class UpdateLayer (m :: * -> *) x where
  -- | The type for the gradient for this layer.
  --   Unit if there isn't a gradient to pass back.
  type Gradient x :: *
  -- | Update a layer with its gradient and learning parameters
  runUpdate      :: LearningParameters -> x -> Gradient x -> m x

-- | Class for a layer. All layers implement this, however, they don't
--   need to implement it for all shapes, only ones which are appropriate.
class UpdateLayer m x => Layer (m :: * -> *) x (i :: Shape) (o :: Shape) where
  -- | Used in training and scoring. Take the input from the previous
  --   layer, and give the output from this layer.
  runForwards    :: x -> S' i -> m (S' o)
  -- | Back propagate a step. Takes the current layer, the input that the
  --   layer gave from the input and the back propagated derivatives from
  --   the layer above.
  --   Returns the gradient layer and the derivatives to push back further.
  runBackards    :: x -> S' i -> S' o -> m (Gradient x, S' i)

-- | Type of a network.
--   The [Shape] type specifies the shapes of data passed between the layers.
--   Could be considered to be a heterogeneous list of layers which are able to
--   transform the data shapes of the network.
data Network :: (* -> *) -> [Shape] -> * where
    O     :: (Show x, Layer m x i o, KnownShape o, KnownShape i)
          => !x
          -> Network m '[i, o]
    (:~>) :: (Show x, Layer m x i h, KnownShape h, KnownShape i)
          => !x
          -> !(Network m (h ': hs))
          -> Network m (i ': h ': hs)
infixr 5 :~>

instance Show (Network m h) where
  show (O a) = "O " ++ show a
  show (i :~> o) = show i ++ "\n:~>\n" ++ show o
