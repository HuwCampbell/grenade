{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE RecordWildCards       #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeSynonymInstances  #-}
{-# LANGUAGE FlexibleInstances     #-}
module Grenade.Layers.FullyConnected (
    FullyConnected (..)
  , FullyConnected' (..)
  , AFullyConnected (..)
  , randomFullyConnected
  ) where

import           Control.Monad.Random hiding (fromList, lift)

import           Data.Proxy
import           Data.Serialize
import           Data.Singletons.TypeLits
import           Data.Array.Accelerate hiding (Shape, fromIntegral)

import qualified Numeric.LinearAlgebra as LA
import           Numeric.LinearAlgebra.Static

import           Grenade.Core
import qualified Grenade.Core.Accelerate as A

import           Grenade.Layers.Internal.Update
import           Grenade.Layers.Internal.Update.Accelerate

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
    let (newBias, newBiasMomentum)    = decendVector learningRate learningMomentum learningRegulariser oldBias biasGradient oldBiasMomentum
        (newActivations, newMomentum) = decendMatrix learningRate learningMomentum learningRegulariser oldActivations activationGradient oldMomentum
    in FullyConnected (FullyConnected' newBias newActivations) (FullyConnected' newBiasMomentum newMomentum)

  createRandom = randomFullyConnected

instance (KnownNat i, KnownNat o) => Layer (FullyConnected i o) ('D1 i) ('D1 o) where
  type Tape (FullyConnected i o) ('D1 i) ('D1 o) = S ('D1 i)
  -- Do a matrix vector multiplication and return the result.
  runForwards (FullyConnected (FullyConnected' wB wN) _) (S1D v) = (S1D v, S1D (wB + wN #> v))

  -- Run a backpropogation step for a full connected layer.
  runBackwards (FullyConnected (FullyConnected' _ wN) _) (S1D x) (S1D dEdy) =
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
        wN = uniformSample s2 (-1) 1
        bm = konst 0
        mm = konst 0
    return $ FullyConnected (FullyConnected' wB wN) (FullyConnected' bm mm)

data AFullyConnected (i :: Nat) (o :: Nat) = AFullyConnected
  (Acc (Vector Double))
  (Acc (Array DIM2 Double))
  (Acc (Vector Double))
  (Acc (Array DIM2 Double))

instance (KnownNat i, KnownNat o) => A.Accelerable (FullyConnected i o) (AFullyConnected i o) where
  toAccel (FullyConnected (FullyConnected' b a) (FullyConnected' bM m)) =
    AFullyConnected
      (use $ A.fromVector b)
      (use $ A.fromMatrix a)
      (use $ A.fromVector bM)
      (use $ A.fromMatrix m)

instance (KnownNat i, KnownNat o) => A.UpdateLayer (FullyConnected i o) (AFullyConnected i o) where

  type Gradient (AFullyConnected i o) = (Acc (Vector Double), Acc (Array DIM2 Double))

  runUpdate
    params
    (AFullyConnected oldBias oldActivations oldBiasMomentum oldMomentum)
    (biasGradient, activationGradient) =
    let (newBias, newBiasMomentum) :: (Acc (Array DIM1 Double), Acc (Array DIM1 Double)) = unlift $ descend params oldBias biasGradient oldBiasMomentum
        (newActivations, newMomentum) :: (Acc (Array DIM2 Double), Acc (Array DIM2 Double)) = unlift $ descend params oldActivations activationGradient oldMomentum
    in AFullyConnected newBias newActivations newBiasMomentum newMomentum


instance (KnownNat i, KnownNat o) => A.Layer (FullyConnected i o) (AFullyConnected i o) DIM1 DIM1 where

  type Tape (AFullyConnected i o) DIM1 DIM1 = Acc (Vector Double)

  runForwards (AFullyConnected wB wN _ _) v = (v, Data.Array.Accelerate.zipWith (+) wB (wN A.#> v))
  runBackwards (AFullyConnected _ wN _ _) x dEdy =
    let wB'  = dEdy
        mm'  = dEdy `A.outer` x
        -- calcluate derivatives for next step
        dWs  = transpose wN A.#> dEdy
    in ((wB', mm'), dWs)
