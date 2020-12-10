{-# LANGUAGE BangPatterns          #-}
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
{-# LANGUAGE Strict                #-}
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
import           Data.Maybe                     (fromMaybe)
import           Data.Reflection                (reifyNat)
import qualified Data.Vector.Storable           as V
import           GHC.Generics                   (Generic)
import           GHC.TypeLits
import           System.Random.MWC              hiding (create)
#if MIN_VERSION_singletons(2,6,0)
import           Data.Singletons.TypeLits       (SNat (..))
#endif
import           Data.List                      (foldl')
import           Data.Proxy
import           Data.Serialize
import           Data.Singletons
import           Data.Singletons.Prelude.Num    ((%*))
import qualified Numeric.LinearAlgebra          as LA
import           Numeric.LinearAlgebra.Static   hiding (zipWithVector)

import           Control.Monad                  (void)
import           Foreign.Storable               (sizeOf)
import           System.IO.Unsafe               (unsafePerformIO)

import           Grenade.Core
import           Grenade.Dynamic
import           Grenade.Dynamic.Internal.Build
import           Grenade.Layers.Internal.BLAS
import           Grenade.Layers.Internal.Memory
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
    !UUID
    !(Int, Int)         -- ^ Input, output
    !(V.Vector RealNum) -- ^ Bias, Temporary vector of same size
    !(V.Vector RealNum) -- ^ Activations
  deriving (Generic)

instance Show (FullyConnected' i o) where
  show FullyConnectedBLAS{}    = "FullyConnectedBLAS"
  show FullyConnectedHMatrix{} = "FullyConnectedHMatrix"

instance NFData (FullyConnected' i o) where
  rnf (FullyConnectedHMatrix b w)      = rnf b `seq` rnf w
  rnf (FullyConnectedBLAS !uuid !io !b !w) = rnf uuid `seq` rnf io `seq` rnf b `seq` rnf w


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
  runUpdate opt@OptSGD {} x@(FullyConnected (FullyConnectedBLAS uuid io@(i,o) oldBiasH oldActivationsH) store) (FullyConnectedBLAS _ _ biasGradient activationGradient) =
      let oldBias = oldBiasH
          oldActivations = oldActivationsH
          (oldMBias, oldMActivations) = case getData opt x store of -- In the first periods until the store is filled newData is called, which will generate FullyConnectedHMatrix instances!
            FullyConnectedBLAS _ _ oldMBias' oldMActivations' -> (oldMBias', oldMActivations')
            FullyConnectedHMatrix oldMBias' oldMActivations' -> (extract oldMBias', extractM oldMActivations')
          VectorResultSGDV newBias newMBias               = descendVectorV opt (VectorValuesSGDV oldBias biasGradient oldMBias)
          MatrixResultSGDV newActivations newMActivations = descendMatrixV opt (MatrixValuesSGDV oldActivations activationGradient oldMActivations)
          newStore = setData opt x store (FullyConnectedBLAS uuid io newMBias newMActivations)
      in  -- releaseTmpVectors uuid `seq`
      FullyConnected (FullyConnectedBLAS uuid io newBias newActivations) newStore
    where extractM mat = (\(S2DV vec) -> vec) . fromS2D $ S2D mat
  runUpdate opt@OptAdam {} x@(FullyConnected (FullyConnectedBLAS uuid io@(i,o) oldBiasH oldActivationsH) store) (FullyConnectedBLAS _ _ biasGradient activationGradient) =
      let oldBias = oldBiasH
          oldActivations = oldActivationsH
          (oldMBias, oldMActivations, oldVBias, oldVActivations) = case getData opt x store of -- In the first periods until the store is filled newData is called, which will generate FullyConnectedHMatrix instances!
            [FullyConnectedBLAS _ _ oldMBias' oldMActivations', FullyConnectedBLAS _ _ oldVBias' oldVActivations'] -> (oldMBias', oldMActivations', oldVBias', oldVActivations')
            [FullyConnectedHMatrix oldMBias' oldMActivations', FullyConnectedHMatrix oldVBias' oldVActivations'] -> (extract oldMBias', extractM oldMActivations', extract oldVBias', extractM oldVActivations')
            [FullyConnectedBLAS _ _ oldMBias' oldMActivations', FullyConnectedHMatrix oldVBias' oldVActivations'] -> (oldMBias', oldMActivations', extract oldVBias', extractM oldVActivations')
            xs -> error $ "unexpected data in ListStore in FullyConnected BLAS implementation: " ++ show xs
          VectorResultAdamV newBias newMBias newVBias                      = descendVectorV opt (VectorValuesAdamV (getStep store) oldBias biasGradient oldMBias oldVBias)
          MatrixResultAdamV newActivations newMActivations newVActivations = descendMatrixV opt (MatrixValuesAdamV (getStep store) oldActivations activationGradient oldMActivations oldVActivations)
          newStore = setData opt x store [FullyConnectedBLAS uuid io newMBias newMActivations, FullyConnectedBLAS uuid io newVBias newVActivations]
      in -- releaseTmpVectors uuid `seq`
        FullyConnected (FullyConnectedBLAS uuid io newBias newActivations) newStore
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
  mapGradient f (FullyConnectedBLAS uuid io bias activations) = FullyConnectedBLAS uuid io (mapVector f bias) (mapVector f activations)
  squaredSums (FullyConnectedHMatrix bias activations) = [sumV . squareV $ bias, sumM . squareM $ activations]
  squaredSums (FullyConnectedBLAS _ _ bias activations) = [V.sum . mapVector (^(2::Int)) $ bias, V.sum . mapVector (^(2::Int)) $ activations]


runForward :: forall i o. (KnownNat i, KnownNat o) => FullyConnected i o -> S ('D1 i) -> (Tape (FullyConnected i o) ('D1 i) ('D1 o), S ('D1 o))
runForward (FullyConnected (FullyConnectedHMatrix wB wN) _) (S1D v) = (S1D v, S1D (wB + wN #> v))
runForward (FullyConnected (FullyConnectedBLAS uuid _ wB wN) _) (S1DV v) =
  let -- !out' = withTempVector uuid (V.length wB) (memCopyVectorFromTo wB >=> matXVec BlasNoTranspose wN v 1.0)
      inp = unsafeMemCopyVectorFromTo wB (createVectorUnsafe (V.length wB))
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
   in -- trace ("ff inp dEdy: " ++ show dEdy)
      -- trace ("ff inp x: " ++ show x)
      -- undefined $
  (FullyConnectedHMatrix wB' mm', S1D dWs)
runBackward (FullyConnected (FullyConnectedBLAS uuid io@(i, o) _ wN) _) (S1DV x) (S1DV dEdy) =
  let !mm' = unsafePerformIO $ outerV dEdy x
      dWsInp = createVectorUnsafe i
      !dWs' = unsafePerformIO $ matXVec BlasTranspose wN dEdy 0 dWsInp
   in -- mm' `seq` dWs' `seq`
    -- (if checkVectors mm' mmCheck then id else trace ("dEdy: " ++ show dEdy ++ "\nx: " ++ show x ++ "\nmmCheck : " ++ show mmCheck ++ "\nmm' : " ++ show mm' ++ "\nmat: " ++ show mat) undefined)
     -- (if checkVecs dWs' mmCheck2 then id else trace ("wN': " ++ show wN) trace ("dEdy : " ++ show dEdy) trace ("dWs:   " ++ show dWs) trace ("\nL i o: " ++ show (tr (matrix $ V.toList wN) :: L i o)) trace ("R o: " ++ show (vector (V.toList dEdy) :: R o)) undefined)
    force mm' `seq` force dWs' `seq` (FullyConnectedBLAS uuid io dEdy mm', S1DV dWs')
runBackward l x dEdy = runBackward l x (toLayerShape x dEdy)


instance (KnownNat i, KnownNat o, KnownNat (i * o)) => Layer (FullyConnected i o) ('D1 i) ('D1 o) where
  type Tape (FullyConnected i o) ('D1 i) ('D1 o) = S ('D1 i) -- V.Vector RealNum
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
  put (FullyConnectedBLAS uuid io b w) = do
    put (1 :: Int)
    put uuid
    put io
    putListOf put . V.toList $ b
    putListOf put . V.toList $ w
  get = do
    (nr :: Int) <- get
    case nr of
      0 -> do
        let f = fromIntegral $ natVal (Proxy :: Proxy i)
        b <- maybe (fail "Vector of incorrect size") return . create . LA.fromList =<< getListOf get
        k <- maybe (fail "Vector of incorrect size") return . create . LA.reshape f . LA.fromList =<< getListOf get
        return $ FullyConnectedHMatrix b k
      1 -> do
        uuid <- get
        io <- get
        b <- V.fromList <$> getListOf get
        w <- V.fromList <$> getListOf get
        return $ FullyConnectedBLAS uuid io b w
      _ -> error $ "Unexpected nr in get in Serialize of FullyConnected' " ++ show nr


instance (KnownNat i, KnownNat o, KnownNat (i*o)) => RandomLayer (FullyConnected i o) where
  createRandomWith = randomFullyConnected


randomFullyConnected :: forall m i o . (PrimBase m, KnownNat i, KnownNat o, KnownNat (i*o))
                     => NetworkInitSettings -> Gen (PrimState m) -> m (FullyConnected i o)
randomFullyConnected (NetworkInitSettings m HMatrix _) gen = do
  wN <- getRandomMatrix i o m gen
  wB <- getRandomVector i o m gen
  return $!! FullyConnected (FullyConnectedHMatrix wB wN) mkListStore
  where i = natVal (Proxy :: Proxy i)
        o = natVal (Proxy :: Proxy o)
randomFullyConnected (NetworkInitSettings m BLAS _) gen = do

  wB <- getRandomVectorV i o o' m gen
  wN <- getRandomVectorV i o (i' * o') m gen
  return $!! FullyConnected (FullyConnectedBLAS newUUID (i', o') wB wN) mkListStore
  where
    i = natVal (Proxy :: Proxy i)
    i' = fromIntegral i
    o = natVal (Proxy :: Proxy o)
    o' = fromIntegral o

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
  zipVectorsWithInPlaceReplSnd f (FullyConnected w1 store1) (FullyConnected w2 store2)= FullyConnected (zipVectorsWithInPlaceReplSnd f w1 w2) (zipVectorsWithInPlaceReplSnd f store1 store2)

instance (KnownNat i, KnownNat o) => GNum (FullyConnected' i o) where
  s |* FullyConnectedHMatrix b w = FullyConnectedHMatrix (dvmap (fromRational s *) b) (dmmap (fromRational s *) w)
  s |* FullyConnectedBLAS uuid io b w = FullyConnectedBLAS uuid io (mapVector (fromRational s *) b) (mapVector (fromRational s *) w)
  FullyConnectedHMatrix b1 w1 |+ FullyConnectedHMatrix b2 w2 = FullyConnectedHMatrix (b1 + b2) (w1 + w2)
  FullyConnectedBLAS uuid io b1 w1 |+ FullyConnectedBLAS _ _ b2 w2 = FullyConnectedBLAS uuid io (zipWithVector (+) b2 b1) (zipWithVector (+) w2 w1)
  x |+ y = error $ "Cannot add different network types in |+ in FullyConnected: " ++ show (x, y)
  zipVectorsWithInPlaceReplSnd f (FullyConnectedBLAS _ _ b1 w1) (FullyConnectedBLAS uuid io b2 w2) =
    FullyConnectedBLAS uuid io (zipWithVectorInPlaceSnd f b1 b2) (zipWithVectorInPlaceSnd f w1 w2)
  -- zipVectorsWithInPlaceReplSnd f (FullyConnectedHMatrix b1 w1) (FullyConnectedHMatrix b2 w2) = FullyConnectedHMatrix (zipWithVector f b1 b2) w2

  zipVectorsWithInPlaceReplSnd _ _ _ = error "zipVectorsWithInPlaceReplSnd only works with BLAS CPU backend. See the NetworkInitSettings."
  sumG xs@(FullyConnectedBLAS uuid io _ _:_) =
    force $ FullyConnectedBLAS uuid io (foldl1 (+) bs) (foldl1 (+) ws)
    -- (foldl' add bs' bs `using` rparWith rdeepseq) `seq` (foldl' add ws' ws `using` rparWith rdeepseq) `seq` FullyConnectedBLAS uuid io bs' ws'
    where
      add acc x = zipWithVectorInPlaceSnd (+) x acc
      (bs, ws) = unzip $ map (\(FullyConnectedBLAS _ _ b w) -> (b, w)) xs
  sumG xs = foldl1 (|+) xs
      -- !bs' = unsafeMemZero $ createVectorUnsafe (V.length $ head bs)
      -- !ws' = unsafeMemZero $ createVectorUnsafe (V.length $ head ws)
  sumG _ = error "SumG only works with BLAS CPU backend. See the NetworkInitSettings to switch CPU backends."
