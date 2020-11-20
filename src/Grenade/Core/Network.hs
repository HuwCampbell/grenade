{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE CPP                   #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE InstanceSigs          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PolyKinds             #-}
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
  , FoldableGradient (..)

  , l2Norm
  , clipByGlobalNorm
  , runNetwork
  , runGradient
  , applyUpdate
  , randomNetwork
  , randomNetworkInitWith
  ) where

import           Control.DeepSeq
import           Control.Monad.IO.Class
import           Control.Monad.Primitive          (PrimBase, PrimState)
import           Control.Parallel.Strategies
import           Data.Default
import           Data.Serialize
import           Data.Singletons
import           Data.Singletons.Prelude
import qualified Data.Vector.Storable             as V
import           GHC.TypeLits                     (KnownNat)
import           Numeric.LinearAlgebra.Static
import           System.Random.MWC

#if MIN_VERSION_base(4,9,0)
import           Data.Kind                        (Type)
#endif

import           Grenade.Core.Layer
import           Grenade.Core.NetworkInitSettings
import           Grenade.Core.NetworkSettings
import           Grenade.Core.Optimizer
import           Grenade.Core.Shape
import           Grenade.Layers.Internal.CUDA     (setCudaTriggerSize)
import           Grenade.Types

-- | Type of a network.
--
--   The @[*]@ type specifies the types of the layers.
--
--   The @[Shape]@ type specifies the shapes of data passed between the layers.
--
--   Can be considered to be a heterogeneous list of layers which are able to
--   transform the data shapes of the network.
data Network :: [Type] -> [Shape] -> Type where
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
  show (x :~> xs) = show x ++ " ~> " ++ show xs

instance NFData (Network '[] '[ i]) where
  rnf NNil = ()
instance (NFData x, NFData (Network xs rs)) => NFData (Network (x ': xs) (i ': rs)) where
  rnf ((!x) :~> (!xs)) = rnf x `seq` rnf xs


-- | Gradient of a network.
--
--   Parameterised on the layers of the network.
data Gradients :: [Type] -> Type where
   GNil  :: Gradients '[]

   (:/>) :: UpdateLayer x
         => !(Gradient x)
         -> !(Gradients xs)
         -> Gradients (x ': xs)

instance NFData (Gradients '[]) where
  rnf GNil = ()
instance (NFData (Gradient x), NFData (Gradients xs)) => NFData (Gradients (x ': xs)) where
  rnf (g :/> gs) = rnf g `seq` rnf gs


instance Serialize (Gradients '[]) where
  put GNil = put ()
  get = return GNil
instance (UpdateLayer x, Serialize (Gradient x), Serialize (Gradients xs)) => Serialize (Gradients (x ': xs)) where
  put (g :/> gs) = put g >> put gs
  get = (:/>) <$> get <*> get


-- | Wegnert Tape of a network.
--
--   Parameterised on the layers and shapes of the network.
data Tapes :: [Type] -> [Shape] -> Type where
   TNil  :: SingI i
         => Tapes '[] '[i]

   (:\>) :: (SingI i, SingI h, Layer x i h)
         => !(Tape x i h)
         -> !(Tapes xs (h ': hs))
         -> Tapes (x ': xs) (i ': h ': hs)

instance NFData (Tapes '[] '[i]) where
  rnf TNil       = ()

instance (NFData (Tape x i h), NFData (Tapes xs (h ': hs))) => NFData (Tapes (x ': xs) (i ': h ': hs)) where
  rnf (t :\> ts) = rnf t `seq` rnf ts

-- | Running a network forwards with some input data.
--
--   This gives the output, and the Wengert tape required for back
--   propagation.
runNetwork :: forall layers shapes.
              Network layers shapes
           -> S (Head shapes)
           -> (Tapes layers shapes, S (Last shapes))
runNetwork = go
    where
  go  :: forall js ss. (Last js ~ Last shapes)
      => Network ss js
      -> S (Head js)
      -> (Tapes ss js, S (Last js))
  go (layer :~> n) !x =
    let (tape, forward) =

          runForwards layer x
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
applyUpdate :: Optimizer opt
            -> Network layers shapes
            -> Gradients layers
            -> Network layers shapes
applyUpdate rate (layer :~> rest) (gradient :/> grest) =
  let layer' = runUpdate rate layer gradient
      rest' = applyUpdate rate rest grest `using` rpar
   in layer' :~> rest'
applyUpdate _ NNil GNil = NNil

-- | Apply network settings across the network.
applySettingsUpdate :: NetworkSettings -> Network layers shapes -> Network layers shapes
applySettingsUpdate settings (layer :~> rest) =
  let layer' = runSettingsUpdate settings layer
      layers' = applySettingsUpdate settings rest `using` rpar
   in layer' :~> layers'
applySettingsUpdate _ NNil = NNil


-- | A network can easily be created by hand with (:~>), but an easy way to
--   initialise a random network is with the @randomNetworkWith@ function.
class CreatableNetwork (xs :: [Type]) (ss :: [Shape])
  -- | Create a network with randomly initialised weights.
  --
  --   Calls to this function will not compile if the type of the neural
  --   network is not sound.
  where
  randomNetworkWith :: PrimBase m => NetworkInitSettings -> Gen (PrimState m) -> m (Network xs ss)

-- | Create a random network using uniform distribution.
randomNetwork :: (MonadIO m, CreatableNetwork xs ss) => m (Network xs ss)
randomNetwork = randomNetworkInitWith def

-- | Create a random network using the specified weight initialization method.
randomNetworkInitWith :: (MonadIO m, CreatableNetwork xs ss) => NetworkInitSettings -> m (Network xs ss)
randomNetworkInitWith m = liftIO $ withSystemRandom . asGenST $ \gen -> randomNetworkWith m gen


instance SingI i => CreatableNetwork '[] '[i] where
  randomNetworkWith initCfg  _ = setCudaTriggerSize (gpuTriggerSize initCfg) >> return NNil

instance (SingI i, SingI o, Layer x i o, RandomLayer x, CreatableNetwork xs (o ': rs)) => CreatableNetwork (x ': xs) (i ': o ': rs) where
  randomNetworkWith m gen = (:~>) <$> createRandomWith m gen <*> randomNetworkWith m gen

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
instance UpdateLayer (Network sublayers subshapes) where
  type Gradient (Network sublayers subshapes) = Gradients sublayers
  runUpdate = applyUpdate
  runSettingsUpdate = applySettingsUpdate

instance FoldableGradient (Gradients '[]) where
  mapGradient _ GNil = GNil
  squaredSums GNil = []

instance (FoldableGradient (Gradient x), FoldableGradient (Gradients xs)) => FoldableGradient (Gradients (x ': xs)) where
  mapGradient f (x :/> xs) =
    let x' = mapGradient f x
        xs' = mapGradient f xs
     in x' :/> xs'
  squaredSums (x :/> xs) = squaredSums x ++ squaredSums xs

-- | Get the L2 Norm of a Foldable Gradient.
l2Norm :: (FoldableGradient x) => x -> RealNum
l2Norm grad = sqrt (sum $ squaredSums grad)

-- | Clip the gradients by the global norm.
clipByGlobalNorm :: (FoldableGradient (Gradients xs)) => RealNum -> Gradients xs -> Gradients xs
clipByGlobalNorm c grads =
  let divisor = sqrt $ sum $ squaredSums grads
   in if divisor > c
        then mapGradient (* (c / divisor)) grads
        else grads


instance CreatableNetwork sublayers subshapes => RandomLayer (Network sublayers subshapes) where
  createRandomWith = randomNetworkWith


-- | Ultimate composition.
--
--   This allows a complete network to be treated as a layer in a larger network.
instance (i ~ (Head subshapes), o ~ (Last subshapes)) => Layer (Network sublayers subshapes) i o where
  type Tape (Network sublayers subshapes) i o = Tapes sublayers subshapes
  runForwards  = runNetwork
  runBackwards = runGradient


--------------------------------------------------

-- | Grenade Num class.
--
-- This allows for instance scalar multiplication of the weights, which is useful for slowly adapting networks, e.g. NN'
-- <- \tau * NN' + (1-\tau) * NN. Or one could sum up some gradients in parallel and apply them at once: @applyUpdate lp
-- net $ foldl1 (|+) ...@aq.
class GNum a where
  (|*) :: Rational -> a -> a
  (|+) :: a -> a -> a
  zipVectorsWithInPlaceReplSnd :: (Double -> Double -> Double) -> a -> a -> a
  sumG :: [a] -> a
  -- default sumG :: [a] -> a
  -- sumG [] = error "sumG called on empty list"

infixl 7 |*
infixr 5 |+

instance (SingI i) => GNum (Network '[] '[ i]) where
  _ |* NNil = NNil
  _ |+ NNil = NNil
  zipVectorsWithInPlaceReplSnd _ _ NNil = NNil
  sumG _ = NNil

instance (SingI i, SingI o, Layer x i o, NFData x, NFData (Network xs (o ': rs)), GNum x, GNum (Network xs (o ': rs))) => GNum (Network (x ': xs) (i ': o ': rs)) where
  s |* (x :~> xs) =
    let x' = (s |* x)
        xs' = (s |* xs) `using` rparWith rdeepseq
     in x' :~> xs'
  (x :~> xs) |+ (y :~> ys) =
    let x' = (x |+ y)
        xs' = (xs |+ ys) `using` rparWith rdeepseq
     in x' :~> xs'
  zipVectorsWithInPlaceReplSnd f (x :~> xs) (y :~> ys) =
    let x' = zipVectorsWithInPlaceReplSnd f x y
        xs' = zipVectorsWithInPlaceReplSnd f xs ys `using` rparWith rdeepseq
     in x' :~> xs'
  sumG xs = sumG (map (\(l :~> _) -> l) xs) :~> sumG (map (\(_ :~> ls) -> ls) xs) `using` rparWith rdeepseq

instance GNum (Gradients '[]) where
  _ |* GNil = GNil
  _ |+ GNil = GNil
  zipVectorsWithInPlaceReplSnd _ _ GNil = GNil
  sumG _ = GNil

instance (GNum a) => GNum [a] where
  r |* xs = fmap (r |*) xs
  xs |+ ys = zipWith (|+) xs ys
  zipVectorsWithInPlaceReplSnd f xs ys = zipWith (zipVectorsWithInPlaceReplSnd f) xs ys
  sumG xs = map sumG xs

instance (UpdateLayer x, GNum (Gradient x), GNum (Gradients xs), NFData (Gradient x), NFData (Gradients xs)) => GNum (Gradients (x ': xs)) where
  s |* (x :/> xs) =
    let x' = (s |* x)
        xs' = (s |* xs) `using` rparWith rdeepseq
     in x' :/> xs'
  (x :/> xs) |+ (y :/> ys) =
    let x' = (x |+ y)
        xs' = (xs |+ ys) `using` rparWith rdeepseq
     in x' :/> xs'
  zipVectorsWithInPlaceReplSnd f (x :/> xs) (y :/> ys) =
    let x' = zipVectorsWithInPlaceReplSnd f x y
        xs' = zipVectorsWithInPlaceReplSnd f xs ys `using` rparWith rdeepseq
     in x' :/> xs'
  sumG xs = sumG (map (\(x :/> _) -> x) xs) :/> sumG (map (\(_ :/> xs') -> xs') xs) `using` rparWith rdeepseq


instance GNum () where
  _ |* () = ()
  _ |+ () = ()
  zipVectorsWithInPlaceReplSnd _ _ () = ()
  sumG _ = ()

instance (GNum a, GNum b) => GNum (a, b) where
  s |* (a, b) = (s |* a, s |* b)
  (a1, b1) |+ (a2, b2) = (a1 |+ a2, b1 |+ b2)
  zipVectorsWithInPlaceReplSnd f (a1, b1) (a2, b2) = (zipVectorsWithInPlaceReplSnd f a1 a2, zipVectorsWithInPlaceReplSnd f b1 b2)
  sumG xs = (sumG (map fst xs), sumG (map snd xs))

instance (KnownNat m) => GNum (R m) where
  s |* vec = dvmap (fromRational s *) vec
  (|+) = (+)
  zipVectorsWithInPlaceReplSnd _ _ _ = error "zipVectorsWithInPlaceReplSnd not implemented for HMatrix CPU-backGrad"
  sumG xs = sum xs


instance (KnownNat m, KnownNat n) => GNum (L m n) where
  s |* mat = dmmap (fromRational s *) mat
  (|+) = (+)
  zipVectorsWithInPlaceReplSnd _ _ _ = error "zipVectorsWithInPlaceReplSnd not implemented for HMatrix CPU-backGrad"
  sumG xs = sum xs
