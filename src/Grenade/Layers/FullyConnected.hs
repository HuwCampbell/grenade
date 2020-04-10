{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE DeriveAnyClass        #-}
{-# LANGUAGE DeriveGeneric         #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE InstanceSigs          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards       #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE UndecidableInstances  #-}
module Grenade.Layers.FullyConnected (
    FullyConnected (..)
  , FullyConnected' (..)
  , randomFullyConnected
  ) where

import           Control.DeepSeq
import           Control.Monad.Primitive        (PrimBase, PrimState)
import           GHC.Generics                   (Generic)
import           GHC.Generics                   (Generic)
import           GHC.TypeLits
import           System.Random.MWC              hiding (create)


import           Data.Proxy
import           Data.Serialize

import qualified Numeric.LinearAlgebra          as LA
import           Numeric.LinearAlgebra.Static

import           Grenade.Core

import           Grenade.Layers.Internal.Update

-- | A basic fully connected (or inner product) neural network layer.
data FullyConnected i o = FullyConnected
                        !(FullyConnected' i o)   -- Neuron weights
                        !(FullyConnected' i o)   -- Neuron momentum
                        deriving (Generic, NFData)

data FullyConnected' i o = FullyConnected'
                         !(R o)   -- Bias
                         !(L o i) -- Activations
                        deriving (Generic, NFData)

instance Show (FullyConnected i o) where
  show FullyConnected {} = "FullyConnected"


instance (KnownNat i, KnownNat o) => UpdateLayer (FullyConnected i o) where
  type Gradient (FullyConnected i o) = (FullyConnected' i o)

  runUpdate LearningParameters {..} (FullyConnected (FullyConnected' oldBias oldActivations) (FullyConnected' oldBiasMomentum oldMomentum)) (FullyConnected' biasGradient activationGradient) =
    let (newBias, newBiasMomentum)    = descendVector learningRate learningMomentum learningRegulariser oldBias biasGradient oldBiasMomentum
        (newActivations, newMomentum) = descendMatrix learningRate learningMomentum learningRegulariser oldActivations activationGradient oldMomentum
    in FullyConnected (FullyConnected' newBias newActivations) (FullyConnected' newBiasMomentum newMomentum)

instance (KnownNat i, KnownNat o) => Layer (FullyConnected i o) ('D1 i) ('D1 o) where
  type Tape (FullyConnected i o) ('D1 i) ('D1 o) = R i
  -- Do a matrix vector multiplication and return the result.
  runForwards (FullyConnected (FullyConnected' wB wN) _) (S1D v) = (v, S1D (wB + wN #> v))

  -- Run a backpropogation step for a full connected layer.
  runBackwards (FullyConnected (FullyConnected' _ wN) _) x (S1D dEdy) =
          let wB'  = dEdy
              mm'  = dEdy `outer` x
              -- calcluate derivatives for next step
              dWs  = tr wN #> dEdy
          in  (FullyConnected' wB' mm', S1D dWs)

instance (KnownNat i, KnownNat o) => Serialize (FullyConnected i o) where
  put (FullyConnected (FullyConnected' b w) _) = do
    putListOf put . LA.toList . extract $ b
    putListOf put . LA.toList . LA.flatten . extract $ w

  get = do
      let f  = fromIntegral $ natVal (Proxy :: Proxy i)
      b     <- maybe (fail "Vector of incorrect size") return . create . LA.fromList =<< getListOf get
      k     <- maybe (fail "Vector of incorrect size") return . create . LA.reshape f . LA.fromList =<< getListOf get
      let bm = konst 0
      let mm = konst 0
      return $ FullyConnected (FullyConnected' b k) (FullyConnected' bm mm)

instance (KnownNat i, KnownNat o, KnownNat (i*o)) => RandomLayer (FullyConnected i o) where

  createRandomWith = randomFullyConnected


randomFullyConnected :: forall m i o . (PrimBase m, KnownNat i, KnownNat o, KnownNat (i*o))
                     => WeightInitMethod -> Gen (PrimState m) -> m (FullyConnected i o)
randomFullyConnected m gen = do
  wN <- getRandomMatrix i o m gen
  wB <- getRandomVector i o m gen
  let bm = konst 0
      mm = konst 0
  return $ FullyConnected (FullyConnected' wB wN) (FullyConnected' bm mm)
  where i = natVal (Proxy :: Proxy i)
        o = natVal (Proxy :: Proxy o)


-------------------- GNum instances --------------------

instance (KnownNat i, KnownNat o) => GNum (FullyConnected i o) where
  s |* FullyConnected i o = FullyConnected (s |* i) o
  FullyConnected i o |+ FullyConnected i2 o2 = FullyConnected (0.5 |* (i |+ i2)) (0.5 |* (o |+ o2))
  gFromRational r = FullyConnected (gFromRational r) (gFromRational 0)

instance (KnownNat i, KnownNat o) => GNum (FullyConnected' i o) where
  s |* FullyConnected' i o = FullyConnected' (fromRational s * i) o
  FullyConnected' i o |+ FullyConnected' i2 o2 = FullyConnected' (0.5 * (i + i2)) (0.5 * (o + o2))
  gFromRational r = FullyConnected' (fromRational r) (fromRational r)

