{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE InstanceSigs          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards       #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TypeOperators         #-}
module Grenade.Layers.FullyConnected (
    FullyConnected (..)
  , FullyConnected' (..)
  , randomFullyConnected
  ) where

import           Control.Monad.Random

import           Data.Proxy
import           Data.Serialize
import           Data.Singletons.TypeLits

import qualified Numeric.LinearAlgebra          as LA
import           Numeric.LinearAlgebra.Static

import           Grenade.Core

import           Grenade.Layers.Internal.Update

-- | A basic fully connected (or inner product) neural network layer.
data FullyConnected i o = FullyConnected
                        !(FullyConnected' i o)   -- Neuron weights
                        !(FullyConnected' i o)   -- Neuron momentum

data FullyConnected' i o = FullyConnected'
                         !(R o)   -- Bias
                         !(L o i) -- Activations

instance Show (FullyConnected i o) where
  show FullyConnected {} = "FullyConnected"

instance (KnownNat i, KnownNat o) => UpdateLayer (FullyConnected i o) where
  type Gradient (FullyConnected i o) = (FullyConnected' i o)

  runUpdate LearningParameters {..} (FullyConnected (FullyConnected' oldBias oldActivations) (FullyConnected' oldBiasMomentum oldMomentum)) (FullyConnected' biasGradient activationGradient) =
    let (newBias, newBiasMomentum)    = descendVector learningRate learningMomentum learningRegulariser oldBias biasGradient oldBiasMomentum
        (newActivations, newMomentum) = descendMatrix learningRate learningMomentum learningRegulariser oldActivations activationGradient oldMomentum
    in FullyConnected (FullyConnected' newBias newActivations) (FullyConnected' newBiasMomentum newMomentum)

  createRandom = randomFullyConnected


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

randomFullyConnected :: (MonadRandom m, KnownNat i, KnownNat o)
                     => m (FullyConnected i o)
randomFullyConnected = do
    s1    <- getRandom
    s2    <- getRandom
    let wB = randomVector  s1 Uniform * 2 - 1
        wN = 1/5000 * uniformSample s2 (-1) 1
        bm = konst 0
        mm = konst 0
    return $ FullyConnected (FullyConnected' wB wN) (FullyConnected' bm mm)


-------------------- Num,Fractional,NMult instances --------------------

-- | Num and Fractional instance of Layer data type for calculating with networks
-- (slowly adapt target network, e.g. as in arXiv: 1509.02971)
instance (KnownNat i,KnownNat o) => Num (FullyConnected i o) where
  FullyConnected i1 o1 + FullyConnected i2 o2 = FullyConnected (i1+i2) (o1+o2)
  FullyConnected i1 o1 * FullyConnected i2 o2 = FullyConnected (i1*i2) (o1*o2)
  FullyConnected i1 o1 - FullyConnected i2 o2 = FullyConnected (i1-i2) (o1-o2)
  abs (FullyConnected i o) = FullyConnected (abs i) (abs o)
  signum (FullyConnected i o) = FullyConnected (signum i) (signum o)
  fromInteger v = FullyConnected (fromInteger v) 0

instance (KnownNat i, KnownNat o) => Fractional (FullyConnected i o) where
  FullyConnected i1 o1 / FullyConnected i2 o2 = FullyConnected (i1/i2) (o1/o2)
  fromRational v = FullyConnected (fromRational v) 0


instance (KnownNat i, KnownNat o) => NMult (FullyConnected i o) where
  s |* FullyConnected i o = FullyConnected (s |* i) o

instance (KnownNat i, KnownNat o) => NMult (FullyConnected' i o) where
  s |* FullyConnected' i o = FullyConnected' (fromRational s * i) (fromRational s * o)


-- | Num and Fractional instance of gradient (for minibatches/batch upgrades)
instance (KnownNat i, KnownNat o) => Num (FullyConnected' i o) where
  FullyConnected' i1 o1 + FullyConnected' i2 o2 = FullyConnected' (i1+i2) (o1+o2)
  FullyConnected' i1 o1 * FullyConnected' i2 o2 = FullyConnected' (i1*i2) (o1*o2)
  FullyConnected' i1 o1 - FullyConnected' i2 o2 = FullyConnected' (i1-i2) (o1-o2)
  abs (FullyConnected' i1 o1) = FullyConnected' (abs i1) (abs o1)
  signum (FullyConnected' i1 o1) = FullyConnected' (signum i1) (signum o1)
  fromInteger v = FullyConnected' (fromInteger v) 0

instance (KnownNat i, KnownNat o) => Fractional (FullyConnected' i o) where
  FullyConnected' i1 o1 / FullyConnected' i2 o2 = FullyConnected' (i1/i2) (o1/o2)
  fromRational v = FullyConnected' (fromRational v) 0


-- v :: (KnownNat x, x~4) => R x
-- v = fromRational nr
--   where nr = fromRational $ toRational (0.001 :: Double)


-- m :: (KnownNat x, KnownNat y, x~4, y~2) => L x y
-- m = matrix [1..8]

-- m2 = fromRational (toRational (2.0 :: Double)) * m
