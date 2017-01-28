{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE FlexibleInstances     #-}
{-|
Module      : Grenade.Core.Network
Description : Core definition a simple neural etwork
Copyright   : (c) Huw Campbell, 2016-2017
License     : BSD2
Stability   : experimental

This module defines the core data type for the simplest
Neural network we support.

-}
module Grenade.Core.Network (
    Layer (..)
  , Network (..)
  , UpdateLayer (..)
  , LearningParameters (..)
  , Gradients (..)
  , CreatableNetwork (..)
  ) where

import           Control.Monad.Random (MonadRandom)
import           Data.List ( foldl' )
import           Data.Singletons

import           Data.Serialize

import           Grenade.Core.Shape

-- | Learning parameters for stochastic gradient descent.
data LearningParameters = LearningParameters {
    learningRate :: Double
  , learningMomentum :: Double
  , learningRegulariser :: Double
  } deriving (Eq, Show)

-- | Class for updating a layer. All layers implement this, and it is
--   shape independent.
class UpdateLayer x where
  {-# MINIMAL runUpdate, createRandom #-}
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

-- | Class for a layer. All layers implement this, however, they don't
--   need to implement it for all shapes, only ones which are appropriate.
class UpdateLayer x => Layer x (i :: Shape) (o :: Shape) where
  -- | The Wengert tape for this layer. Includes all that is required
  --   to generate the back propagated gradients efficiently. As a
  --   default, `S i` is fine.
  type Tape x i o :: *
  -- | Used in training and scoring. Take the input from the previous
  --   layer, and give the output from this layer.
  runForwards    :: x -> S i -> (Tape x i o, S o)
  -- | Back propagate a step. Takes the current layer, the input that the
  --   layer gave from the input and the back propagated derivatives from
  --   the layer above.
  --   Returns the gradient layer and the derivatives to push back further.
  runBackwards   :: x -> Tape x i o -> S o -> (Gradient x, S i)

-- | Type of a network.
--
--   The [*] type specifies the types of the layers.
--
--   The [Shape] type specifies the shapes of data passed between the layers.
--
--   Can be considered to be a heterogeneous list of layers which are able to
--   transform the data shapes of the network.
data Network :: [*] -> [Shape] -> * where
    NNil  :: SingI i => Network '[] '[i]
    (:~>) :: (SingI i, SingI h, Layer x i h) => !x -> !(Network xs (h ': hs)) -> Network (x ': xs) (i ': h ': hs)
infixr 5 :~>

instance Show (Network '[] '[i]) where
  show NNil = "NNil"
instance (Show x, Show (Network xs rs)) => Show (Network (x ': xs) (i ': rs)) where
  show (x :~> xs) = show x ++ "\n~>\n" ++ show xs

-- | Gradients of a network.
--   Parameterised on the layers of a Network.
data Gradients :: [*] -> * where
   GNil  :: Gradients '[]
   (:/>) :: UpdateLayer x => Gradient x -> Gradients xs -> Gradients (x ': xs)

-- | A network can easily be created by hand with (:~>), but an easy way to initialise a random
--   network is with the randomNetwork.
class CreatableNetwork (xs :: [*]) (ss :: [Shape]) where
  -- | Create a network of the types requested
  randomNetwork :: MonadRandom m => m (Network xs ss)

instance SingI i => CreatableNetwork '[] '[i] where
  randomNetwork = return NNil

instance (SingI i, SingI o, Layer x i o, CreatableNetwork xs (o ': rs)) => CreatableNetwork (x ': xs) (i ': o ': rs) where
  randomNetwork = (:~>) <$> createRandom <*> randomNetwork


-- | Add very simple serialisation to the network
instance SingI i => Serialize (Network '[] '[i]) where
  put NNil = pure ()
  get = return NNil

instance (SingI i, SingI o, Layer x i o, Serialize x, Serialize (Network xs (o ': rs))) => Serialize (Network (x ': xs) (i ': o ': rs)) where
  put (x :~> r) = put x >> put r
  get = (:~>) <$> get <*> get
