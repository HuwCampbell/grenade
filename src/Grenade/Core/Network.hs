{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE UndecidableInstances  #-}
{-|
Module      : Grenade.Core.Network
Description : Core definition of a Neural Network
Copyright   : (c) Huw Campbell, 2016-2017
License     : BSD2
Stability   : experimental

This module defines the core data types and functions
for non-recurrent neural networks.
-}

module Grenade.Core.Network (
    Network (..)
  , CreatableNetwork (..)
  , Gradients (..)
  , Tapes (..)
  , GNum (..)

  , runNetwork
  , runGradient
  , applyUpdate
  , randomNetwork

  ) where

import           Control.DeepSeq
import           Control.Monad.Random              (MonadRandom)

import           Data.Serialize
import           Data.Singletons
import           Data.Singletons.Prelude

import           Grenade.Core.Layer
import           Grenade.Core.LearningParameters
import           Grenade.Core.Shape
import           Grenade.Core.WeightInitialization

-- | Type of a network.
--
--   The @[*]@ type specifies the types of the layers.
--
--   The @[Shape]@ type specifies the shapes of data passed between the layers.
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

instance NFData (Network '[] '[i]) where
  rnf NNil = ()
instance (NFData x, NFData (Network xs rs)) => NFData (Network (x ': xs) (i ': rs)) where
  rnf (x :~> xs) = rnf x `seq` rnf xs


-- | Gradient of a network.
--
--   Parameterised on the layers of the network.
data Gradients :: [*] -> * where
   GNil  :: Gradients '[]

   (:/>) :: UpdateLayer x
         => Gradient x
         -> Gradients xs
         -> Gradients (x ': xs)

instance NFData (Gradients '[]) where
  rnf GNil = ()
instance (NFData (Gradient x), NFData (Gradients xs)) => NFData (Gradients (x ': xs)) where
  rnf (g :/> gs) = rnf g `seq` rnf gs

-- | Wegnert Tape of a network.
--
--   Parameterised on the layers and shapes of the network.
data Tapes :: [*] -> [Shape] -> * where
   TNil  :: SingI i
         => Tapes '[] '[i]

   (:\>) :: (SingI i, SingI h, Layer x i h)
         => !(Tape x i h)
         -> !(Tapes xs (h ': hs))
         -> Tapes (x ': xs) (i ': h ': hs)


-- | Running a network forwards with some input data.
--
--   This gives the output, and the Wengert tape required for back
--   propagation.
runNetwork :: forall layers shapes.
              Network layers shapes
           -> S (Head shapes)
           -> (Tapes layers shapes, S (Last shapes))
runNetwork =
  go
    where
  go  :: forall js ss. (Last js ~ Last shapes)
      => Network ss js
      -> S (Head js)
      -> (Tapes ss js, S (Last js))
  go (layer :~> n) !x =
    let (tape, forward) = runForwards layer x
        (tapes, answer) = go n forward
    in  (tape :\> tapes, answer)

  go NNil !x
      = (TNil, x)


-- | Running a loss gradient back through the network.
--
--   This requires a Wengert tape, generated with the appropriate input
--   for the loss.
--
--   Gives the gradients for the layer, and the gradient across the
--   input (which may not be required).
runGradient :: forall layers shapes.
               Network layers shapes
            -> Tapes layers shapes
            -> S (Last shapes)
            -> (Gradients layers, S (Head shapes))
runGradient net tapes o =
  go net tapes
    where
  go  :: forall js ss. (Last js ~ Last shapes)
      => Network ss js
      -> Tapes ss js
      -> (Gradients ss, S (Head js))
  go (layer :~> n) (tape :\> nt) =
    let (gradients, feed)  = go n nt
        (layer', backGrad) = runBackwards layer tape feed
    in  (layer' :/> gradients, backGrad)

  go NNil TNil
      = (GNil, o)


-- | Apply one step of stochastic gradient descent across the network.
applyUpdate :: LearningParameters
            -> Network layers shapes
            -> Gradients layers
            -> Network layers shapes
applyUpdate rate (layer :~> rest) (gradient :/> grest)
  = runUpdate rate layer gradient :~> applyUpdate rate rest grest

applyUpdate _ NNil GNil
  = NNil

-- | A network can easily be created by hand with (:~>), but an easy way to
--   initialise a random network is with the randomNetwork.
class CreatableNetwork (xs :: [*]) (ss :: [Shape]) where
  -- | Create a network with randomly initialised weights.
  --
  --   Calls to this function will not compile if the type of the neural
  --   network is not sound.
  randomNetworkWith :: MonadRandom m => WeightInitMethod -> m (Network xs ss)

-- | Create a random network using uniform distribution.
randomNetwork :: (MonadRandom m, CreatableNetwork xs ss) => m (Network xs ss)
randomNetwork = randomNetworkWith UniformInit

instance SingI i => CreatableNetwork '[] '[i] where
  randomNetworkWith _ = return NNil

instance (SingI i, SingI o, Layer x i o, RandomLayer x, CreatableNetwork xs (o ': rs)) => CreatableNetwork (x ': xs) (i ': o ': rs) where
  randomNetworkWith m = (:~>) <$> createRandomWith m <*> randomNetworkWith m

-- | Add very simple serialisation to the network
instance SingI i => Serialize (Network '[] '[i]) where
  put NNil = pure ()
  get = return NNil

instance (SingI i, SingI o, Layer x i o, Serialize x, Serialize (Network xs (o ': rs))) => Serialize (Network (x ': xs) (i ': o ': rs)) where
  put (x :~> r) = put x >> put r
  get = (:~>) <$> get <*> get


-- | Ultimate composition.
--
--   This allows a complete network to be treated as a layer in a larger network.
instance CreatableNetwork sublayers subshapes => UpdateLayer (Network sublayers subshapes) where
  type Gradient (Network sublayers subshapes) = Gradients sublayers
  runUpdate    = applyUpdate

instance  CreatableNetwork sublayers subshapes => RandomLayer (Network sublayers subshapes) where
  createRandomWith = randomNetworkWith

-- | Ultimate composition.
--
--   This allows a complete network to be treated as a layer in a larger network.
instance (CreatableNetwork sublayers subshapes, i ~ (Head subshapes), o ~ (Last subshapes)) => Layer (Network sublayers subshapes) i o where
  type Tape (Network sublayers subshapes) i o = Tapes sublayers subshapes
  runForwards  = runNetwork
  runBackwards = runGradient


-- | Grenade Num class.
--
-- This allows for instance scalar multiplication of the weights, which is useful for slowly adapting networks, e.g. NN'
-- <- \tau * NN' + (1-\tau) * NN. Or one could sum up some gradients in parallel and apply them at once after
-- normalizing.
class GNum a where
  (|*) :: Rational -> a -> a
  (|+) :: a -> a -> a
  gFromRational :: Rational -> a

infixl 7 |*
infixr 5 |+

instance (SingI i) => GNum (Network '[] '[i]) where
  _ |* NNil = NNil
  _ |+ NNil = NNil
  gFromRational _ = NNil

instance (SingI i, SingI o, Layer x i o, GNum x, GNum (Network xs (o ': rs))) => GNum (Network (x ': xs) (i ': o ': rs)) where
  s |* (x :~> xs) = (s |* x) :~> (s |* xs)
  (x :~> xs) |+ (y :~> ys) = (x |+ y) :~> (xs |+ ys)
  gFromRational r = gFromRational r :~> gFromRational r

instance GNum (Gradients '[]) where
  _ |* GNil = GNil
  _ |+ GNil = GNil
  gFromRational _ = GNil

instance (UpdateLayer x, GNum (Gradient x), GNum (Gradients xs)) => GNum (Gradients (x ': xs)) where
  s |* (x :/> xs) = (s |* x) :/> (s |* xs)
  (x :/> xs) |+ (y :/> ys) = (x |+ y) :/> (xs |+ ys)
  gFromRational r = gFromRational r :/> gFromRational r

instance GNum () where
  _ |* () = ()
  _ |+ () = ()
  gFromRational _ = ()

instance (GNum a, GNum b) => GNum (a,b) where
  s |* (a,b) = (s|*a,s|*b)
  (a1,b1) |+ (a2,b2) = (a1|+a2,b1|+b2)
  gFromRational v = (gFromRational v, gFromRational v)


