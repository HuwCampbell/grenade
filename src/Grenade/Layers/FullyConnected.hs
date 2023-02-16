{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE CPP                   #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE DeriveAnyClass        #-}
{-# LANGUAGE DeriveGeneric         #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE FlexibleInstances     #-}
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
import           Control.Monad
import           Control.Monad.Primitive        (PrimBase, PrimState)
import           Control.Parallel.Strategies
import           Data.List                      (foldl')
import           Data.Maybe                     (fromMaybe)
import           Data.Proxy
import           Data.Reflection                (reifyNat)
import           Data.Serialize
import qualified Data.Vector.Storable           as V
import           GHC.Generics                   (Generic)

import           Data.Singletons                hiding ((*))
import           GHC.TypeLits
import           GHC.TypeLits.KnownNat
import           GHC.TypeLits.Singletons
import           Prelude.Singletons             ((%*))

import qualified Numeric.LinearAlgebra          as LA
import           Numeric.LinearAlgebra.Static   hiding (zipWithVector)
import           System.Random.MWC              hiding (create)
import           Text.Printf

import           Control.Monad                  (void)
import           Foreign.Storable               (peekElemOff, pokeElemOff, sizeOf)
import           System.IO.Unsafe               (unsafePerformIO)

import           Grenade.Core
import           Grenade.Dynamic
import           Grenade.Dynamic.Internal.Build
import           Grenade.Layers.Internal.BLAS
import           Grenade.Layers.Internal.CUDA
import           Grenade.Layers.Internal.Update
import           Grenade.Types
import           Grenade.Utils.Conversion
import           Grenade.Utils.LinearAlgebra
import           Grenade.Utils.ListStore
import           Grenade.Utils.Vector


import           Debug.Trace

-- | A basic fully connected (or inner product) neural network layer.
data FullyConnected i o = FullyConnected
                        !(FullyConnected' i o)             -- Neuron weights
                        !(ListStore (FullyConnected' i o)) -- momentum store
                        deriving (Generic)

instance NFData (FullyConnected i o) where
  rnf (FullyConnected w store) = rnf w `seq` rnf store

instance Show (FullyConnected i o) where
  show FullyConnected {} = "FullyConnected"


-- | How to store the data .
data FullyConnected' i o
  = FullyConnectedHMatrix
      !(R o)   -- ^ Bias
      !(L o i) -- ^ Activations
  | FullyConnectedBLAS
    !(Int, Int)         -- ^ Input, output
    !(V.Vector RealNum) -- ^ Bias, Temporary vector of same size
    !(V.Vector RealNum) -- ^ Activations
  deriving (Generic)

instance Show (FullyConnected' i o) where
  show FullyConnectedBLAS{}    = "FullyConnectedBLAS"
  show FullyConnectedHMatrix{} = "FullyConnectedHMatrix"

instance NFData (FullyConnected' i o) where
  rnf (FullyConnectedHMatrix b w)    = rnf b `seq` rnf w
  rnf (FullyConnectedBLAS !io !b !w) = rnf io `seq` rnf b `seq` rnf w


instance (KnownNat i, KnownNat o, KnownNat (i * o)) => UpdateLayer (FullyConnected i o) where
  type Gradient (FullyConnected i o) = (FullyConnected' i o)
  type MomentumStore (FullyConnected i o) = ListStore (FullyConnected' i o)
  runUpdate opt@OptSGD {} x@(FullyConnected (FullyConnectedHMatrix oldBias oldActivations) store) (FullyConnectedHMatrix biasGradient activationGradient) =
    let (FullyConnectedHMatrix oldBiasMomentum oldMomentum) = getData opt x store
        VectorResultSGD newBias newBiasMomentum = descendVector opt (VectorValuesSGD oldBias biasGradient oldBiasMomentum)
        MatrixResultSGD newActivations newMomentum = descendMatrix opt (MatrixValuesSGD oldActivations activationGradient oldMomentum)
        newStore = setData opt x store (FullyConnectedHMatrix newBiasMomentum newMomentum)
     in FullyConnected (FullyConnectedHMatrix newBias newActivations) newStore
  runUpdate opt@OptAdam {} x@(FullyConnected (FullyConnectedHMatrix oldBias oldActivations) store) (FullyConnectedHMatrix biasGradient activationGradient) =
    let [FullyConnectedHMatrix oldMBias oldMActivations, FullyConnectedHMatrix oldVBias oldVActivations] = getData opt x store
        VectorResultAdam newBias newMBias newVBias = descendVector opt (VectorValuesAdam (getStep store) oldBias biasGradient oldMBias oldVBias)
        MatrixResultAdam newActivations newMActivations newVActivations =
          descendMatrix opt (MatrixValuesAdam (getStep store) oldActivations activationGradient oldMActivations oldVActivations)
        newStore = setData opt x store [FullyConnectedHMatrix newMBias newMActivations, FullyConnectedHMatrix newVBias newVActivations]
     in FullyConnected (FullyConnectedHMatrix newBias newActivations) newStore
  runUpdate opt@OptSGD {} x@(FullyConnected (FullyConnectedBLAS io@(i,o) oldBiasH oldActivationsH) store) (FullyConnectedBLAS _ biasGradient activationGradient) =
      let oldBias = oldBiasH
          oldActivations = oldActivationsH
          (oldMBias, oldMActivations) = case getData opt x store of -- In the first periods until the store is filled newData is called, which will generate FullyConnectedHMatrix instances!
            FullyConnectedBLAS _ oldMBias' oldMActivations'  -> (oldMBias', oldMActivations')
            FullyConnectedHMatrix oldMBias' oldMActivations' -> (extract oldMBias', extractM oldMActivations')
          VectorResultSGDV newBias newMBias               = descendVectorV opt (VectorValuesSGDV oldBias biasGradient oldMBias)
          MatrixResultSGDV newActivations newMActivations = descendMatrixV opt (MatrixValuesSGDV oldActivations activationGradient oldMActivations)
          newStore = setData opt x store (FullyConnectedBLAS io newMBias newMActivations)
      in FullyConnected (FullyConnectedBLAS io newBias newActivations) newStore
    where extractM mat = (\(S2DV vec) -> vec) . fromS2D $ S2D mat
  runUpdate opt@OptAdam {} x@(FullyConnected (FullyConnectedBLAS io@(i,o) oldBiasH oldActivationsH) store) (FullyConnectedBLAS _ biasGradient activationGradient) =
      let oldBias = oldBiasH
          oldActivations = oldActivationsH
          (oldMBias, oldMActivations, oldVBias, oldVActivations) = case getData opt x store of -- In the first periods until the store is filled newData is called, which will generate FullyConnectedHMatrix instances!
            [FullyConnectedBLAS _ oldMBias' oldMActivations', FullyConnectedBLAS _ oldVBias' oldVActivations'] -> (oldMBias', oldMActivations', oldVBias', oldVActivations')
            [FullyConnectedHMatrix oldMBias' oldMActivations', FullyConnectedHMatrix oldVBias' oldVActivations'] -> (extract oldMBias', extractM oldMActivations', extract oldVBias', extractM oldVActivations')
            [FullyConnectedBLAS _ oldMBias' oldMActivations', FullyConnectedHMatrix oldVBias' oldVActivations'] -> (oldMBias', oldMActivations', extract oldVBias', extractM oldVActivations')
            xs -> error $ "unexpected data in ListStore in FullyConnected BLAS implementation: " ++ show xs
          VectorResultAdamV newBias newMBias newVBias                      = descendVectorV opt (VectorValuesAdamV (getStep store) oldBias biasGradient oldMBias oldVBias)
          MatrixResultAdamV newActivations newMActivations newVActivations = descendMatrixV opt (MatrixValuesAdamV (getStep store) oldActivations activationGradient oldMActivations oldVActivations)
          newStore = setData opt x store [FullyConnectedBLAS io newMBias newMActivations, FullyConnectedBLAS io newVBias newVActivations]
      in FullyConnected (FullyConnectedBLAS io newBias newActivations) newStore
    where extractM mat = (\(S2DV vec) -> vec) . fromS2D $ S2D mat
  runUpdate opt (FullyConnected layer _) _ = error $ "Unexpected input in runUpdate in FullyConnected layer. Optimizer" ++ show opt ++ ". Layer: " ++ show layer

instance (KnownNat i, KnownNat o, KnownNat (i * o)) => LayerOptimizerData (FullyConnected i o) (Optimizer 'SGD) where
  type MomentumDataType (FullyConnected i o) (Optimizer 'SGD) = FullyConnected' i o
  getData opt x store = head $ getListStore opt x store
  setData opt x store = setListStore opt x store . return
  newData _ _ = FullyConnectedHMatrix (konst 0) (konst 0)

instance (KnownNat i, KnownNat o, KnownNat (i * o)) => LayerOptimizerData (FullyConnected i o) (Optimizer 'Adam) where
  type MomentumDataType (FullyConnected i o) (Optimizer 'Adam) = FullyConnected' i o
  type MomentumExpOptResult (FullyConnected i o) (Optimizer 'Adam) = [FullyConnected' i o]
  getData = getListStore
  setData = setListStore
  newData _ _ = FullyConnectedHMatrix (konst 0) (konst 0)


instance (KnownNat i, KnownNat o) => FoldableGradient (FullyConnected' i o) where
  mapGradient f (FullyConnectedHMatrix bias activations) = FullyConnectedHMatrix (dvmap f bias) (dmmap f activations)
  mapGradient f (FullyConnectedBLAS io bias activations) = FullyConnectedBLAS io (mapVector f bias) (mapVector f activations)
  squaredSums (FullyConnectedHMatrix bias activations) = [sumV . squareV $ bias, sumM . squareM $ activations]
  squaredSums (FullyConnectedBLAS _ bias activations)  = [V.sum . mapVector (^(2::Int)) $ bias, V.sum . mapVector (^(2::Int)) $ activations]


runForward :: forall i o. (KnownNat i, KnownNat o) => FullyConnected i o -> S ('D1 i) -> (Tape (FullyConnected i o) ('D1 i) ('D1 o), S ('D1 o))
runForward (FullyConnected (FullyConnectedHMatrix wB wN) _) (S1D v) = (S1D v, S1D (wB + wN #> v))
runForward (FullyConnected (FullyConnectedBLAS _ wB wN) _) (S1DV v) =
  let inp = unsafeMemCopyVectorFromTo wB (createVectorUnsafe (V.length wB))
      !out' = unsafePerformIO $ matXVec BlasNoTranspose wN v 1 inp
   in force out' `seq` (S1DV v, S1DV out')
runForward lay@(FullyConnected (FullyConnectedHMatrix _ _) _ ) v@S1DV{} = runForward lay (toS1D v)
runForward lay@(FullyConnected FullyConnectedBLAS{} _) v@S1D{} = runForward lay (fromS1D v)


runBackward :: forall i o . (KnownNat i, KnownNat o) => FullyConnected i o -> Tape (FullyConnected i o) ('D1 i) ('D1 o) -> S ('D1 o) -> (Gradient (FullyConnected i o), S ('D1 i))
runBackward (FullyConnected (FullyConnectedHMatrix _ wN) _) (S1D x) (S1D dEdy) =
  let wB' = dEdy
      mm' = dEdy `outer` x
            -- calcluate derivatives for next step
      dWs = tr wN #> dEdy
   in (FullyConnectedHMatrix wB' mm', S1D dWs)
runBackward (FullyConnected (FullyConnectedBLAS io@(i, o) _ wN) _) (S1DV x) (S1DV dEdy) =
  let !mm' = unsafePerformIO $ outerV dEdy x
      dWsInp = createVectorUnsafe i
      !dWs' = unsafePerformIO $ matXVec BlasTranspose wN dEdy 0 dWsInp
   in force mm' `seq` force dWs' `seq` (FullyConnectedBLAS io dEdy mm', S1DV dWs')
runBackward l x dEdy = runBackward l x (toLayerShape x dEdy)


instance (KnownNat i, KnownNat o, KnownNat (i * o)) => Layer (FullyConnected i o) ('D1 i) ('D1 o) where
  type Tape (FullyConnected i o) ('D1 i) ('D1 o) = S ('D1 i)
  runForwards = runForward   -- Do a matrix vector multiplication and return the result.
  runBackwards = runBackward -- Run a backpropogation step for a full connected layer.


instance (KnownNat i, KnownNat o) => Serialize (FullyConnected i o) where
  put (FullyConnected w ms) = put w >> put ms
  get = FullyConnected <$> get <*> get

instance (KnownNat i, KnownNat o) => Serialize (FullyConnected' i o) where
  put (FullyConnectedHMatrix b w) = do
    put (0 :: Int)
    putListOf put . LA.toList . extract $ b
    putListOf put . LA.toList . LA.flatten . extract $ w
  put (FullyConnectedBLAS io b w) = do
    put (1 :: Int)
    put io
    putListOf put . V.toList $ b
    putListOf put . V.toList $ w
  get = do
    (nr :: Int) <- get
    case nr of
      0 -> do
        let f = fromIntegral $ GHC.TypeLits.natVal (Proxy :: Proxy i)
        b <- maybe (fail "Vector of incorrect size") return . create . LA.fromList =<< getListOf get
        k <- maybe (fail "Vector of incorrect size") return . create . LA.reshape f . LA.fromList =<< getListOf get
        return $ FullyConnectedHMatrix b k
      1 -> do
        io <- get
        b <- V.fromList <$> getListOf get
        w <- V.fromList <$> getListOf get
        return $ FullyConnectedBLAS io b w
      _ -> error $ "Unexpected nr in get in Serialize of FullyConnected' " ++ show nr


instance (KnownNat i, KnownNat o, KnownNat (i * o)) => RandomLayer (FullyConnected i o) where
  createRandomWith = randomFullyConnected


randomFullyConnected ::
     forall m i o. (PrimBase m, KnownNat i, KnownNat o, KnownNat (i * o))
  => NetworkInitSettings
  -> Gen (PrimState m)
  -> m (FullyConnected i o)
randomFullyConnected (NetworkInitSettings m HMatrix _) gen = do
  wN <- getRandomMatrix i o m gen
  wB <- getRandomVector i o m gen
  return $!! FullyConnected (FullyConnectedHMatrix wB wN) mkListStore
  where i = GHC.TypeLits.natVal (Proxy :: Proxy i)
        o = GHC.TypeLits.natVal (Proxy :: Proxy o)
randomFullyConnected (NetworkInitSettings m BLAS _) gen = do

  wB <- getRandomVectorV i o o' m gen
  wN <- getRandomVectorV i o (i' * o') m gen
  return $!! FullyConnected (FullyConnectedBLAS (i', o') wB wN) mkListStore
  where
    i = GHC.TypeLits.natVal (Proxy :: Proxy i)
    i' = fromIntegral i
    o = GHC.TypeLits.natVal (Proxy :: Proxy o)
    o' = fromIntegral o

-------------------- DynamicNetwork instance --------------------

instance (KnownNat i, KnownNat o) => FromDynamicLayer (FullyConnected i o) where
  fromDynamicLayer _ _ _ = SpecNetLayer $ SpecFullyConnected (GHC.TypeLits.natVal (Proxy :: Proxy i)) (GHC.TypeLits.natVal (Proxy :: Proxy o))

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

instance (KnownNat i, KnownNat o) => GNum (FullyConnected' i o) where
  s |* FullyConnectedHMatrix b w = FullyConnectedHMatrix (dvmap (fromRational s *) b) (dmmap (fromRational s *) w)
  s |* FullyConnectedBLAS io b w = FullyConnectedBLAS io (mapVector (fromRational s *) b) (mapVector (fromRational s *) w)
  FullyConnectedHMatrix b1 w1 |+ FullyConnectedHMatrix b2 w2 = FullyConnectedHMatrix (b1 + b2) (w1 + w2)
  FullyConnectedBLAS io b1 w1 |+ FullyConnectedBLAS _ b2 w2  = FullyConnectedBLAS io (zipWithVector (+) b2 b1) (zipWithVector (+) w2 w1)
  x |+ y                                                     = error $ "Cannot add different network types in |+ in FullyConnected: " ++ show (x, y)
  sumG xs@(FullyConnectedBLAS io _ _:_) = FullyConnectedBLAS io bs' ws'
    where
      (bs, ws) = unzip $ map (\(FullyConnectedBLAS _ b w) -> (b, w)) xs
      bs' = sumVectors bs `using` rparWith rdeepseq
      ws' = sumVectors ws `using` rparWith rdeepseq
  sumG xs = foldl1 (|+) xs
