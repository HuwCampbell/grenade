{-# LANGUAGE CPP                   #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE ScopedTypeVariables   #-}
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

#if __GLASGOW_HASKELL__ < 800
{-# OPTIONS_GHC -fno-warn-incomplete-patterns #-}
#endif

module Grenade.Core.Network (
    Network (..)
  , Gradients (..)
  , Tapes (..)
  , CreatableNetwork (..)

  , applyUpdate
  ) where

import           Control.Monad.Random ( MonadRandom )

import           Data.Singletons
import           Data.Singletons.Prelude
import           Data.Serialize

import           Grenade.Core.Layer
import           Grenade.Core.LearningParameters
import           Grenade.Core.Shape

-- | Type of a network.
--
--   The [*] type specifies the types of the layers.
--
--   The [Shape] type specifies the shapes of data passed between the layers.
--
--   Can be considered to be a heterogeneous list of layers which are able to
--   transform the data shapes of the network.
data Network :: [*] -> [Shape] -> * where
    NNil  :: SingI i
          => Network '[] '[i]

    (:~>) :: (SingI i, SingI h, Layer x i h)
          => !x
          -> !(Network xs (h ': hs))
          -> Network (x ': xs) (i ': h ': hs)
infixr 5 :~>

instance Show (Network '[] '[i]) where
  show NNil = "NNil"
instance (Show x, Show (Network xs rs)) => Show (Network (x ': xs) (i ': rs)) where
  show (x :~> xs) = show x ++ "\n~>\n" ++ show xs


-- | Apply one step of stochastic gradient decent across the network.
applyUpdate :: LearningParameters -> Network ls ss -> Gradients ls -> Network ls ss
applyUpdate _ NNil GNil
  = NNil
applyUpdate rate (layer :~> rest) (gradient :/> grest)
  = runUpdate rate layer gradient :~> applyUpdate rate rest grest



-- | Gradients of a network.
--   Parameterised on the layers of the network.
data Gradients :: [*] -> * where
   GNil  :: Gradients '[]
   (:/>) :: UpdateLayer x => Gradient x -> Gradients xs -> Gradients (x ': xs)

-- | Wegnert Tapes of a network.
--   Parameterised on the layers and shapes of the network.
data Tapes :: [*] -> [Shape] -> * where
   TNil  :: SingI i => Tapes '[] '[i]
   (:\>) :: (SingI i, SingI h, Layer x i h) => !(Tape x i h) -> !(Tapes xs (h ': hs)) -> Tapes (x ': xs) (i ': h ': hs)

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


-- | Ultimate composition.
--
--   This allows a complete network to be treated as a layer in a bigger network.
instance CreatableNetwork sublayers subshapes => UpdateLayer (Network sublayers subshapes) where
  type Gradient (Network sublayers subshapes) = Gradients sublayers
  runUpdate = applyUpdate
  createRandom = randomNetwork

instance (CreatableNetwork sublayers subshapes, i ~ (Head subshapes), o ~ (Last subshapes)) => Layer (Network sublayers subshapes) i o where
  type Tape (Network sublayers subshapes) i o = Tapes sublayers subshapes

  runForwards net i =
    go i net
      where
    go  :: forall js ss. (Last js ~ Last subshapes)
        => S (Head js)          -- ^ input vector
        -> Network ss js -- ^ network to train
        -> (Tapes ss js, S (Last js))
    go !x (layer :~> n) =
      let (tape, forward) = runForwards layer x
          (tapes, answer) = go forward n
      in  (tape :\> tapes, answer)
    go !x NNil
        = (TNil, x)

  runBackwards net tapes o =
    go net tapes
      where
    go  :: forall js ss. (Last js ~ Last subshapes)
        => Network ss js -- ^ network to train
        -> Tapes ss js -- ^ network to train
        -> (Gradients ss, S (Head js))
    go (layer :~> n) (tape :\> nt) =
      let (gradients, feed)  = go n nt
          (layer', backGrad) = runBackwards layer tape feed
      in  (layer' :/> gradients, backGrad)
    go NNil TNil
        = (GNil, o)
