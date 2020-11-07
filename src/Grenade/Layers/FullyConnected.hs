{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE CPP                   #-}
{-# LANGUAGE DataKinds             #-}
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
import           Data.Maybe                     (fromMaybe)
import           Data.Reflection                (reifyNat)
import qualified Data.Vector.Storable           as V
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

import           Control.Monad.ST               (RealWorld)
import           Data.Vector.Storable.Mutable   as VM
import           Foreign.Storable
import           System.IO.Unsafe               (unsafePerformIO)

import           Grenade.Core
import           Grenade.Dynamic
import           Grenade.Dynamic.Internal.Build
import           Grenade.Layers.Internal.CBLAS  as B
import           Grenade.Layers.Internal.Update
import           Grenade.Types
import           Grenade.Utils.LinearAlgebra
import           Grenade.Utils.ListStore

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
  | FullyConnectedHBLAS
    !(Int, Int)                          -- ^ Nr of Inputs/Outputs
    !(V.Vector RealNum)                  -- ^ Bias
    !(V.Vector RealNum)                  -- ^ Activations
    !(V.Vector RealNum,V.Vector RealNum) -- ^ additional temporary matrices that can be used without the need of mallocing data
  deriving (Generic)

instance NFData (FullyConnected' i o) where
  rnf (FullyConnectedHMatrix b w)       = rnf b `seq` rnf w
  rnf (FullyConnectedHBLAS io !_ !_ !_) = rnf io


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
  runUpdate opt@OptAdam {} x@(FullyConnected blas@(FullyConnectedHBLAS io oldBiasH oldActivationsH tmp) store) (FullyConnectedHMatrix biasGradient activationGradient) =
      let oldBias = fromMaybe err . create $ oldBiasH
          oldActivations = matrix . V.toList $ oldActivationsH
          [FullyConnectedHMatrix oldMBias oldMActivations, FullyConnectedHMatrix oldVBias oldVActivations] = getData opt x store
          VectorResultAdam newBias newMBias newVBias = descendVector opt (VectorValuesAdam (getStep store) oldBias biasGradient oldMBias oldVBias)
          MatrixResultAdam newActivations newMActivations newVActivations =
            descendMatrix opt (MatrixValuesAdam (getStep store) oldActivations activationGradient oldMActivations oldVActivations)
          newStore = setData opt x store [FullyConnectedHMatrix newMBias newMActivations, FullyConnectedHMatrix newVBias newVActivations]
      -- V.copy (HBLAS._bufferMutDenseVector oldBiasH) (extract newBias)
      -- V.copy (HBLAS._bufferDenMutMat oldActivationsH) (V.concat $ map extract (toRows newActivations))
      in FullyConnected (FullyConnectedHMatrix newBias newActivations) newStore
    where err = error "cannot create vector/matrix in runUpdate of FullyConnected layer using HBLAS"
  runUpdate opt@OptSGD{} (FullyConnected FullyConnectedHMatrix{} _) _ = error "HBLAS is only implemented for OptAdam in runUpdate for FullyConnected layer"

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
  squaredSums (FullyConnectedHMatrix bias activations) = [sumV . squareV $ bias, sumM . squareM $ activations]


runForward :: (KnownNat i, KnownNat o) => FullyConnected i o -> S (D1 i) -> (Tape (FullyConnected i o) ('D1 i) ('D1 o), S ('D1 o))
runForward (FullyConnected (FullyConnectedHMatrix wB wN) _) (S1D v) = (extract v, S1D (wB + wN #> v))
runForward (FullyConnected (FullyConnectedHMatrix wB wN) _) (S1DV v) = error "unexpected mixing of HBLAS and HMatrix in FullyConnected"
runForward lay@(FullyConnected FullyConnectedHBLAS{} _) (S1D v) = runForward lay (S1DV $ extract v)
runForward (FullyConnected (FullyConnectedHBLAS (i,o) wB wN (tmpInp, tmpOut)) _) (S1DV v) =

    -- inpVecM <- V.thaw v
    -- VM.copy (HBLAS._bufferDenMutMat tmpInp) inpVecM
    -- VM.copy (HBLAS._bufferDenMutMat tmpOut) (HBLAS._bufferMutDenseVector wB)
    let out = wB
    -- HBLAS.dgemm HBLAS.NoTranspose HBLAS.NoTranspose 1.0 1.0 wN tmpInp tmpOut
        out' = trace ("runFoward") dgemmUnsafe BlasRowMajor B.BlasNoTranspose B.BlasNoTranspose (1, i) (i,o)  (1, o) 1.0 v wN 1.0 out
    -- out <- V.freeze (HBLAS._bufferDenMutMat tmpOut)
    in (v, S1DV out')


runBackward :: forall i o . (KnownNat i, KnownNat o) => FullyConnected i o -> Tape (FullyConnected i o) ('D1 i) ('D1 o) -> S ('D1 o) -> (Gradient (FullyConnected i o), S ('D1 i))
runBackward (FullyConnected (FullyConnectedHMatrix _ wN) _) x (S1D dEdy) =
          let wB'  = dEdy
              mm'  = dEdy `outer` vector (V.toList x)
              -- calcluate derivatives for next step
              dWs  = tr wN #> dEdy
          in  (FullyConnectedHMatrix wB' mm', S1D dWs)
runBackward (FullyConnected (FullyConnectedHMatrix _ wN) _) x (S1DV dEdy) = error "unexpected mixing of HBLAS and HMatrix types in FullyConnected"
runBackward lay@(FullyConnected FullyConnectedHBLAS{} _) x (S1D dEdy) = runBackward lay x (S1DV $ extract dEdy)
runBackward (FullyConnected (FullyConnectedHBLAS (i, o) _ wN (tmpInp, tmpOut)) _) x (S1DV dEdy) =
    -- xVec <- V.thaw x
    -- VM.copy (HBLAS._bufferDenMutMat tmpInp) xVec
    -- dEdyVec <- V.thaw dEdy
    -- weights

  -- VM.copy (HBLAS._bufferDenMutMat tmpOut) dEdyVec
  -- mmM <- HBLAS.generateMutableDenseMatrix HBLAS.SRow (o, i) (const 0)
  -- HBLAS.dgemm HBLAS.NoTranspose HBLAS.Transpose 1.0 0.0 tmpInp tmpOut mmM
  -- dEdy * x
  let mm = V.replicate (i*o) 0
      mm' = trace ("runBackward") dgemmUnsafe BlasRowMajor B.BlasTranspose B.BlasTranspose (1, i) (o, 1) (i, o) 1.0 x dEdy 0.0 mm
      mmCheck  = V.concat $ map (V.map realToFrac . extract) $ toColumns $ (vector (V.toList dEdy) :: R o) `outer` (vector (V.toList x) :: R i)

    -- moments
  -- dWsM <- HBLAS.generateMutableDenseMatrix HBLAS.SRow (1, i) (const 0)
      dWs =  V.replicate i 0
  -- HBLAS.dgemm HBLAS.Transpose HBLAS.NoTranspose 1.0 0 wN tmpOut dWsM
      dWs' = trace ("runBackward2") dgemmUnsafe BlasRowMajor B.BlasTranspose B.BlasNoTranspose (o, 1) (o, i)  (1, i) 1.0 dEdy wN 0.0 dWs
    -- derivatives
  -- dWs <- fromMaybe err . create <$> V.freeze (HBLAS._bufferDenMutMat dWsM)
  in (if mm' == mmCheck
     then id
     else trace ("mm' == mmcheck: " ++ show (mm' == mmCheck)) trace (show mm') trace (show mmCheck) undefined)


    force (FullyConnectedHMatrix (fromMaybe err $ create dEdy) (matrix $ V.toList mm'), S1DV dWs')
  where
    err = error "could not create matrix in runBackwards in FullyConnectedHMatrix using HBLAS"

instance (KnownNat i, KnownNat o, KnownNat (i * o)) => Layer (FullyConnected i o) ('D1 i) ('D1 o) where
  type Tape (FullyConnected i o) ('D1 i) ('D1 o) = V.Vector RealNum
  -- Do a matrix vector multiplication and return the result.
  runForwards = runForward
  -- runForwards (FullyConnected (FullyConnectedHMatrix wB wN) _) (S1D v) = (v, S1D (wB + wN #> v))
  -- runForwards lay@(FullyConnected FullyConnectedHBLAS{} _) (S1D v) = go

  --   where go = runForwards lay (S1DV $ extract v)
  -- runForwards (FullyConnected (FullyConnectedHBLAS (i,o) wB wN (tmpInp, tmpOut)) _) (S1DV v) = unsafePerformIO $ do
  --   inpVecM <- V.thaw (extract v)
  --   VM.copy (HBLAS._bufferDenMutMat tmpInp) inpVecM
  --   VM.copy (HBLAS._bufferDenMutMat tmpOut) (HBLAS._bufferMutDenseVector wB)
  --   HBLAS.dgemm HBLAS.NoTranspose HBLAS.NoTranspose 1.0 1.0 wN tmpInp tmpOut
  --   out <- fromMaybe err . create <$> V.freeze (HBLAS._bufferDenMutMat tmpOut)
  --   return (v, S1D out)
  --   where err = error "could not create vector in runForwards in FullyConnected using HBLAS"

  -- Run a backpropogation step for a full connected layer.
  runBackwards = runBackward


instance (KnownNat i, KnownNat o) => Serialize (FullyConnected i o) where
  put (FullyConnected w ms) = put w >> put ms
  get = FullyConnected <$> get <*> get


instance (KnownNat i, KnownNat o) => Serialize (FullyConnected' i o) where
  put (FullyConnectedHMatrix b w) = do
    putListOf put . LA.toList . extract $ b
    putListOf put . LA.toList . LA.flatten . extract $ w
  get = do
      let f  = fromIntegral $ natVal (Proxy :: Proxy i)
      b     <- maybe (fail "Vector of incorrect size") return . create . LA.fromList =<< getListOf get
      k     <- maybe (fail "Vector of incorrect size") return . create . LA.reshape f . LA.fromList =<< getListOf get
      return $ FullyConnectedHMatrix b k


instance (KnownNat i, KnownNat o, KnownNat (i*o)) => RandomLayer (FullyConnected i o) where
  createRandomWith = randomFullyConnected


randomFullyConnected :: forall m i o . (PrimBase m, KnownNat i, KnownNat o, KnownNat (i*o))
                     => NetworkInitSettings -> Gen (PrimState m) -> m (FullyConnected i o)
randomFullyConnected (NetworkInitSettings m HMatrix) gen = do
  wN <- getRandomMatrix i o m gen
  wB <- getRandomVector i o m gen
  return $!! FullyConnected (FullyConnectedHMatrix wB wN) mkListStore
  where i = natVal (Proxy :: Proxy i)
        o = natVal (Proxy :: Proxy o)
randomFullyConnected (NetworkInitSettings m HBLAS) gen = do
  let vm = 1
      vk = 4
      vn = 5
      ma = (vm, vk)
      mb = (vk, vn)
      mc = (vm, vn)
      len (a,b) = a * b

      a = V.fromList (Prelude.replicate (len ma) 3) :: V.Vector Double
      b = V.fromList (Prelude.replicate (len mb) 4) :: V.Vector Double
      c = V.fromList (Prelude.replicate (len mc) 0) :: V.Vector Double

      wB = vector (V.toList c) :: R 4
      wN = matrix (V.toList b) :: L 4 5
      v = vector (V.toList a) :: R 5

      c' = dgemmUnsafe BlasRowMajor B.BlasNoTranspose B.BlasNoTranspose ma mb mc 1.0 a b 1.0 c
      -- mm' = trace ("runBackward") dgemmUnsafe BlasRowMajor B.BlasNoTranspose B.BlasNoTranspose (o, 1) (1, i) (o, i) 1.0 dEdy x 0.0 mm
  -- trace ("c': " ++ show c')
  --   trace ("c': " ++ show (wB + wN #> v))
  --   undefined


      -- wB <- HBLAS.generateMutableDenseVector o' (randsV !!)
      -- wN <- HBLAS.generateMutableDenseMatrix HBLAS.SRow (i', o') (\(x, y) -> randsM !! (x * o' + y))
      -- tmpVecInp <- HBLAS.generateMutableDenseMatrix HBLAS.SRow (1, i') (const 0)
      -- tmpVecOut <- HBLAS.generateMutableDenseMatrix HBLAS.SRow (1, o') (const 0)


  wB <- V.fromList <$> getRandomList i o o' m gen
  wN <- V.fromList <$> getRandomList i o (i' * o') m gen
  let tmpVecInp = V.replicate i' 0
      tmpVecOut = V.replicate o' 0
  return $ FullyConnected (FullyConnectedHBLAS (i', o') wB wN (tmpVecInp, tmpVecOut)) mkListStore
  where
    i = natVal (Proxy :: Proxy i)
    i' = fromIntegral i
    o = natVal (Proxy :: Proxy o)
    o' = fromIntegral o
-- randomFullyConnected (NetworkInitSettings _ cpu) _ = error $ "CPU backend " ++ show cpu ++ " not supported by FullyConnected layer"

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
  s |* FullyConnectedHMatrix b w = FullyConnectedHMatrix (dvmap (fromRational s *) b) (dmmap (fromRational s *) w)
  FullyConnectedHMatrix b1 w1 |+ FullyConnectedHMatrix b2 w2 = FullyConnectedHMatrix (b1 + b2) (w1 + w2)
  gFromRational r = FullyConnectedHMatrix (fromRational r) (fromRational r)
