{-# LANGUAGE CPP                   #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE DeriveAnyClass        #-}
{-# LANGUAGE DeriveGeneric         #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE InstanceSigs          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE UndecidableInstances  #-}
{-# OPTIONS_GHC -Wno-incomplete-uni-patterns #-}

module Grenade.Layers.FullyConnected (
    FullyConnected (..)
  , FullyConnected' (..)
  , randomFullyConnected
  , SpecFullyConnected (..)
  , specFullyConnected
  , fullyConnected
  ) where

import           Control.DeepSeq
import           Control.Monad.Primitive        (PrimBase, PrimState)
import           Data.Reflection                (reifyNat)
import           GHC.Generics                   (Generic)
import           GHC.TypeLits
import           System.Random.MWC              hiding (create)
#if MIN_VERSION_singletons(2,6,0)
import           Data.Singletons.TypeLits       (SNat (..))
#endif
import           Data.Proxy
import           Data.Serialize
import           Data.Singletons
import           Data.Singletons.Prelude.Num    ((%*))

import qualified Numeric.LinearAlgebra          as LA
import           Numeric.LinearAlgebra.Static

import           Grenade.Core
import           Grenade.Dynamic
import           Grenade.Dynamic.Internal.Build
import           Grenade.Layers.Internal.Update
import           Grenade.Utils.ListStore

-- | A basic fully connected (or inner product) neural network layer.
data FullyConnected i o = FullyConnected
                        !(FullyConnected' i o)   -- Neuron weights
                        !(ListStore (FullyConnected' i o))   -- momentum store
                        deriving (Generic)

instance NFData (FullyConnected i o) where
  rnf (FullyConnected w store) = rnf w `seq` rnf store


data FullyConnected' i o = FullyConnected'
                         !(R o)   -- Bias
                         !(L o i) -- Activations
                        deriving (Generic)

instance NFData (FullyConnected' i o) where
  rnf (FullyConnected' b w) = rnf b `seq` rnf w


instance Show (FullyConnected i o) where
  show FullyConnected {} = "FullyConnected"


instance (KnownNat i, KnownNat o, KnownNat (i * o)) => UpdateLayer (FullyConnected i o) where
  type Gradient (FullyConnected i o) = (FullyConnected' i o)
  type MomentumStore (FullyConnected i o) = ListStore (FullyConnected' i o)
  runUpdate opt@OptSGD{} x@(FullyConnected (FullyConnected' oldBias oldActivations) store) (FullyConnected' biasGradient activationGradient) =
    let (FullyConnected' oldBiasMomentum oldMomentum) = getData opt x store
        VectorResultSGD newBias newBiasMomentum = descendVector opt (VectorValuesSGD oldBias biasGradient oldBiasMomentum)
        MatrixResultSGD newActivations newMomentum = descendMatrix opt (MatrixValuesSGD oldActivations activationGradient oldMomentum)
        newStore = setData opt x store (FullyConnected' newBiasMomentum newMomentum)
     in FullyConnected (FullyConnected' newBias newActivations) newStore
  runUpdate opt@OptAdam{} x@(FullyConnected (FullyConnected' oldBias oldActivations) store) (FullyConnected' biasGradient activationGradient) =
    let [FullyConnected' oldMBias oldMActivations, FullyConnected' oldVBias oldVActivations] = getData opt x store
        VectorResultAdam newBias newMBias newVBias = descendVector opt (VectorValuesAdam (getStep store) oldBias biasGradient oldMBias oldVBias)
        MatrixResultAdam newActivations newMActivations newVActivations = descendMatrix opt (MatrixValuesAdam (getStep store) oldActivations activationGradient oldMActivations oldVActivations)
        newStore = setData opt x store [FullyConnected' newMBias newMActivations, FullyConnected' newVBias newVActivations]
    in FullyConnected (FullyConnected' newBias newActivations) newStore

instance (KnownNat i, KnownNat o, KnownNat (i * o)) => LayerOptimizerData (FullyConnected i o) (Optimizer 'SGD) where
  type MomentumDataType (FullyConnected i o) (Optimizer 'SGD) = FullyConnected' i o
  getData opt x store = head $ getListStore opt x store
  setData opt x store = setListStore opt x store . return
  newData _ _ = FullyConnected' (konst 0) (konst 0)

instance (KnownNat i, KnownNat o, KnownNat (i * o)) => LayerOptimizerData (FullyConnected i o) (Optimizer 'Adam) where
  type MomentumDataType (FullyConnected i o) (Optimizer 'Adam) = FullyConnected' i o
  type MomentumExpOptResult (FullyConnected i o) (Optimizer 'Adam) = [FullyConnected' i o]
  getData = getListStore
  setData = setListStore
  newData _ _ = FullyConnected' (konst 0) (konst 0)


instance (KnownNat i, KnownNat o, KnownNat (i * o)) => Layer (FullyConnected i o) ('D1 i) ('D1 o) where
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
  put (FullyConnected w ms) = put w >> put ms
  get = FullyConnected <$> get <*> get


instance (KnownNat i, KnownNat o) => Serialize (FullyConnected' i o) where
  put (FullyConnected' b w) = do
    putListOf put . LA.toList . extract $ b
    putListOf put . LA.toList . LA.flatten . extract $ w
  get = do
      let f  = fromIntegral $ natVal (Proxy :: Proxy i)
      b     <- maybe (fail "Vector of incorrect size") return . create . LA.fromList =<< getListOf get
      k     <- maybe (fail "Vector of incorrect size") return . create . LA.reshape f . LA.fromList =<< getListOf get
      return $ FullyConnected' b k


instance (KnownNat i, KnownNat o, KnownNat (i*o)) => RandomLayer (FullyConnected i o) where
  createRandomWith = randomFullyConnected


randomFullyConnected :: forall m i o . (PrimBase m, KnownNat i, KnownNat o, KnownNat (i*o))
                     => WeightInitMethod -> Gen (PrimState m) -> m (FullyConnected i o)
randomFullyConnected m gen = do
  wN <- getRandomMatrix i o m gen
  wB <- getRandomVector i o m gen
  return $ FullyConnected (FullyConnected' wB wN) mkListStore
  where i = natVal (Proxy :: Proxy i)
        o = natVal (Proxy :: Proxy o)

-------------------- DynamicNetwork instance --------------------

instance (KnownNat i, KnownNat o) => FromDynamicLayer (FullyConnected i o) where
  fromDynamicLayer _ _ _ = SpecNetLayer $ SpecFullyConnected (natVal (Proxy :: Proxy i)) (natVal (Proxy :: Proxy o))

instance ToDynamicLayer SpecFullyConnected where
  toDynamicLayer wInit gen (SpecFullyConnected nrI nrO) =
    reifyNat nrI $ \(pxInp :: (KnownNat i) => Proxy i) ->
      reifyNat nrO $ \(pxOut :: (KnownNat o') => Proxy o') ->
        case singByProxy pxInp %* singByProxy pxOut of
          SNat -> do
            (layer :: FullyConnected i o') <- randomFullyConnected wInit gen
            return $ SpecLayer layer (sing :: Sing ('D1 i)) (sing :: Sing ('D1 o'))

-- | Make a specification of a fully connected layer (see Grenade.Dynamic.Build for a user-interface to specifications).
specFullyConnected :: Integer -> Integer -> SpecNet
specFullyConnected nrI nrO = SpecNetLayer $ SpecFullyConnected nrI nrO


-- | A Fully-connected layer with input dimensions as given in last output layer and output dimensions specified. 1D only!
fullyConnected :: Integer -> BuildM ()
fullyConnected rows = do
  (inRows, _, _) <- buildRequireLastLayerOut Is1D
  buildAddSpec (SpecNetLayer $ SpecFullyConnected inRows rows)
  buildSetLastLayer (rows, 1, 1)


-------------------- GNum instances --------------------

instance (KnownNat i, KnownNat o) => GNum (FullyConnected i o) where
  s |* FullyConnected w store = FullyConnected (s |* w) (s |* store)
  FullyConnected w1 store1 |+ FullyConnected w2 store2 = FullyConnected (w1 |+ w2) (store1 |+ store2)
  gFromRational r = FullyConnected (gFromRational r) mkListStore

instance (KnownNat i, KnownNat o) => GNum (FullyConnected' i o) where
  s |* FullyConnected' b w = FullyConnected' (dvmap (fromRational s *) b) (dmmap (fromRational s *) w)
  FullyConnected' b1 w1 |+ FullyConnected' b2 w2 = FullyConnected' (b1 + b2) (w1 + w2)
  gFromRational r = FullyConnected' (fromRational r) (fromRational r)


