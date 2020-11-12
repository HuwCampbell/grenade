{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE GADTs                    #-}
module Grenade.Layers.Internal.Update (
    descendMatrix
  , descendVector
  , MatrixInputValues (..)
  , MatrixResult (..)
  , VectorInputValues (..)
  , VectorResult (..)
  , descendMatrixV
  , descendVectorV
  , MatrixInputValuesV (..)
  , MatrixResultV (..)
  , VectorInputValuesV (..)
  , VectorResultV (..)
  ) where

import           Control.Parallel.Strategies
import           Data.Maybe                   (fromJust)
import qualified Data.Vector.Storable         as U (unsafeFromForeignPtr0,
                                                    unsafeToForeignPtr0)
import qualified Data.Vector.Storable         as V
import           Foreign                      (mallocForeignPtrArray, withForeignPtr)
import           Foreign.Ptr                  (Ptr)
import           GHC.TypeLits
import           Numeric.LinearAlgebra        (Vector, flatten)
import qualified Numeric.LinearAlgebra.Devel  as U
import           Numeric.LinearAlgebra.Static
import           System.IO.Unsafe             (unsafePerformIO)

import           Grenade.Core.Optimizer
import           Grenade.Types

data MatrixInputValues rows columns
  = MatrixValuesSGD
      !(L rows columns) -- ^ current weights
      !(L rows columns) -- ^ gradients
      !(L rows columns) -- ^ last update (old momentum)
  | MatrixValuesAdam
      !Int              -- ^ Step
      !(L rows columns) -- ^ current weights
      !(L rows columns) -- ^ gradients
      !(L rows columns) -- ^ current m
      !(L rows columns) -- ^ current v

data MatrixInputValuesV
  = MatrixValuesSGDV
      !(V.Vector RealNum) -- ^ current weights
      !(V.Vector RealNum) -- ^ gradients
      !(V.Vector RealNum) -- ^ old momentum
  | MatrixValuesAdamV
      !Int                -- ^ Step
      !(V.Vector RealNum) -- ^ current weights
      !(V.Vector RealNum) -- ^ gradients
      !(V.Vector RealNum) -- ^ current m
      !(V.Vector RealNum) -- ^ current v


data MatrixResult rows columns
  = MatrixResultSGD
      { matrixActivations :: !(L rows columns) -- ^ new activations (weights)
      , matrixMomentum    :: !(L rows columns) -- ^ new momentum
      }
  | MatrixResultAdam
      { matrixActivations :: !(L rows columns) -- ^ new activations (weights)
      , matrixM           :: !(L rows columns) -- ^ new m
      , matrixV           :: !(L rows columns) -- ^ new v
      }

data MatrixResultV
  = MatrixResultSGDV
      { matrixActivationsV :: !(V.Vector RealNum) -- ^ new activations (weights)
      , matrixMV           :: !(V.Vector RealNum) -- ^ new m
      }
  | MatrixResultAdamV
      { matrixActivationsV :: !(V.Vector RealNum) -- ^ new activations (weights)
      , matrixMV           :: !(V.Vector RealNum) -- ^ new m
      , matrixVV           :: !(V.Vector RealNum) -- ^ new v
      }

data VectorInputValues r
  = VectorValuesSGD
      !(R r) -- ^ current weights
      !(R r) -- ^ gradients
      !(R r) -- ^ last update (old momentum)
  | VectorValuesAdam
      !Int   -- ^ Step
      !(R r) -- ^ current weights
      !(R r) -- ^ current gradients
      !(R r) -- ^ current m
      !(R r) -- ^ current v

data VectorInputValuesV
  = VectorValuesSGDV
      !(V.Vector RealNum) -- ^ current weights
      !(V.Vector RealNum) -- ^ current gradients
      !(V.Vector RealNum) -- ^ current m
  | VectorValuesAdamV
      !Int                -- ^ Step
      !(V.Vector RealNum) -- ^ current weights
      !(V.Vector RealNum) -- ^ current gradients
      !(V.Vector RealNum) -- ^ current m
      !(V.Vector RealNum) -- ^ current v

data VectorResult r
  = VectorResultSGD
      { vectorBias     :: !(R r) -- ^ new activations (bias)
      , vectorMomentum :: !(R r) -- ^ new momentum
      }
  | VectorResultAdam
      { vectorBias :: !(R r) -- ^ new activations (bias)
      , vectorM    :: !(R r) -- ^ new m
      , vectorV    :: !(R r) -- ^ new v
      }

data VectorResultV
  = VectorResultSGDV
      { vectorBiasV :: !(V.Vector RealNum) -- ^ new activations (bias)
      , vectorMV    :: !(V.Vector RealNum) -- ^ new m
      }
  | VectorResultAdamV
      { vectorBiasV :: !(V.Vector RealNum) -- ^ new activations (bias)
      , vectorMV    :: !(V.Vector RealNum) -- ^ new m
      , vectorVV    :: !(V.Vector RealNum) -- ^ new v
      }

descendMatrix :: (KnownNat rows, KnownNat columns) => Optimizer o -> MatrixInputValues rows columns -> MatrixResult rows columns
descendMatrix (OptSGD rate momentum regulariser) (MatrixValuesSGD weights gradient lastUpdate) =
  let (rows, cols) = size weights
      len          = rows * cols
      -- Most gradients come in in ColumnMajor,
      -- so we'll transpose here before flattening them
      -- into a vector to prevent a copy.
      --
      -- This gives ~15% speed improvement for LSTMs.
      weights'     = flatten . tr . extract $ weights
      gradient'    = flatten . tr . extract $ gradient
      lastUpdate'  = flatten . tr . extract $ lastUpdate
      (vw, vm)     = descendUnsafeSGD len rate momentum regulariser weights' gradient' lastUpdate'

      -- Note that it's ColumnMajor, as we did a transpose before
      -- using the internal vectors.
      mw           = U.matrixFromVector U.ColumnMajor rows cols vw
      mm           = U.matrixFromVector U.ColumnMajor rows cols vm
  in  MatrixResultSGD (fromJust . create $ mw) (fromJust . create $ mm)
descendMatrix (OptAdam alpha beta1 beta2 epsilon lambda) (MatrixValuesAdam step weights gradient m v) =
  let (rows, cols) = size weights
      len          = rows * cols
      -- Most gradients come in in ColumnMajor,
      -- so we'll transpose here before flattening them
      -- into a vector to prevent a copy.
      --
      -- This gives ~15% speed improvement for LSTMs.
      weights'  = flatten . tr . extract $ weights
      gradient' = flatten . tr . extract $ gradient
      m'        = flatten . tr . extract $ m
      v'        = flatten . tr . extract $ v
      (vw, vm, vv)     = descendUnsafeAdam len step alpha beta1 beta2 epsilon lambda weights' gradient' m' v'

      -- Note that it's ColumnMajor, as we did a transpose before
      -- using the internal vectors.
      mw           = U.matrixFromVector U.ColumnMajor rows cols vw
      mm           = U.matrixFromVector U.ColumnMajor rows cols vm
      mv           = U.matrixFromVector U.ColumnMajor rows cols vv
  in  MatrixResultAdam (fromJust . create $ mw) (fromJust . create $ mm) (fromJust . create $ mv)
descendMatrix opt _ = error $ "optimzer does not match to MatrixInputValues in implementation! Optimizer: " ++ show opt

descendMatrixV :: Optimizer o -> MatrixInputValuesV -> MatrixResultV
descendMatrixV (OptSGD rate momentum regulariser) (MatrixValuesSGDV weights gradient lastUpdate) =
  let len          = V.length weights
      (vw, vm)     = descendUnsafeSGD len rate momentum regulariser weights gradient lastUpdate
  in  MatrixResultSGDV vw vm
descendMatrixV (OptAdam alpha beta1 beta2 epsilon lambda) (MatrixValuesAdamV step weights gradient m v) =
  let len          = V.length weights
      (vw, vm, vv) = descendUnsafeAdam len step alpha beta1 beta2 epsilon lambda weights gradient m v
  in  MatrixResultAdamV vw vm vv
descendMatrixV opt _ = error $ "optimzer does not match to MatrixInputValues in implementation! Optimizer: " ++ show opt

descendVector :: (KnownNat r) => Optimizer o -> VectorInputValues r -> VectorResult r
descendVector (OptSGD rate momentum regulariser) (VectorValuesSGD weights gradient lastUpdate) =
  let len          = size weights
      weights'     = extract weights
      gradient'    = extract gradient
      lastUpdate'  = extract lastUpdate
      (vw, vm)     = descendUnsafeSGD len rate momentum regulariser weights' gradient' lastUpdate'
  in  VectorResultSGD (fromJust $ create vw) (fromJust $ create vm)
descendVector (OptAdam alpha beta1 beta2 epsilon lambda) (VectorValuesAdam step weights gradient m v) =
  let len       = size weights
      weights'  = extract weights
      gradient' = extract gradient
      m'        = extract m
      v'        = extract v
      (vw, vm, vv)     = descendUnsafeAdam len step alpha beta1 beta2 epsilon lambda weights' gradient' m' v'
  in  VectorResultAdam (fromJust $ create vw) (fromJust $ create vm) (fromJust $ create vv)
descendVector opt _ = error $ "optimzer does not match to VectorInputValues in implementation! Optimizer: " ++ show opt

descendVectorV :: Optimizer o -> VectorInputValuesV -> VectorResultV
descendVectorV (OptSGD rate momentum regulariser) (VectorValuesSGDV weights gradient lastUpdate) =
  let len          = V.length weights
      (vw, vm)     = descendUnsafeSGD len rate momentum regulariser weights gradient lastUpdate
  in  VectorResultSGDV vw vm
descendVectorV (OptAdam alpha beta1 beta2 epsilon lambda) (VectorValuesAdamV step weights gradient m v) =
  let len = V.length weights
      (vw, vm, vv) = descendUnsafeAdam len step alpha beta1 beta2 epsilon lambda weights gradient m v
   in VectorResultAdamV vw vm vv
descendVectorV opt _ = error $ "optimzer does not match to VectorInputValues in implementation! Optimizer: " ++ show opt


-- -- | Caching of data
-- type CacheKey = (LookupType, ProxyType, StateFeatures)

-- cacheMVar :: MVar (M.Map CacheKey [Values])
-- cacheMVar = unsafePerformIO $ newMVar mempty
-- {-# NOINLINE cacheMVar #-}

-- emptyCache :: MonadIO m => m ()
-- emptyCache = liftIO $ modifyMVar_ cacheMVar (const mempty)

-- addCache :: (MonadIO m) => CacheKey -> [Values] -> m ()
-- addCache k val = liftIO $ modifyMVar_ cacheMVar (return . M.insert k val)

-- lookupCache :: (MonadIO m) => CacheKey -> m (Maybe [Values])
-- lookupCache k = liftIO $ (M.lookup k =<<) <$> tryReadMVar cacheMVar


-- -- | Get output of function f, if possible from cache according to key (st).
-- cached :: (MonadIO m) => (LookupType, ProxyType, StateFeatures) -> m [Values] -> m [Values]
-- cached st ~f = do
--   c <- lookupCache st
--   case c of
--     Nothing -> do
--       res <- f
--       res `seq` addCache st res
--       return res
--     Just res -> return res

descendUnsafeSGD :: Int -> RealNum -> RealNum -> RealNum -> Vector RealNum -> Vector RealNum -> Vector RealNum -> (Vector RealNum, Vector RealNum)
descendUnsafeSGD len rate momentum regulariser weights gradient lastUpdate =
  unsafePerformIO $ do
    outWPtr <- mallocForeignPtrArray len
    outMPtr <- mallocForeignPtrArray len
    let (wPtr, _) = U.unsafeToForeignPtr0 weights
    let (gPtr, _) = U.unsafeToForeignPtr0 gradient
    let (lPtr, _) = U.unsafeToForeignPtr0 lastUpdate

    withForeignPtr wPtr $ \wPtr' ->
      withForeignPtr gPtr $ \gPtr' ->
        withForeignPtr lPtr $ \lPtr' ->
          withForeignPtr outWPtr $ \outWPtr' ->
            withForeignPtr outMPtr $ \outMPtr' ->
              descend_sgd_cpu len rate momentum regulariser wPtr' gPtr' lPtr' outWPtr' outMPtr'

    return (U.unsafeFromForeignPtr0 outWPtr len, U.unsafeFromForeignPtr0 outMPtr len)
{-# NOINLINE descendUnsafeSGD #-}

descendUnsafeAdam ::
     Int -- Len
  -> Int -- Step
  -> RealNum -- Alpha
  -> RealNum -- Beta1
  -> RealNum -- Beta2
  -> RealNum -- Epsilon
  -> RealNum -- Lambda
  -> Vector RealNum -- Weights
  -> Vector RealNum -- Gradient
  -> Vector RealNum -- M
  -> Vector RealNum -- V
  -> (Vector RealNum, Vector RealNum, Vector RealNum)
descendUnsafeAdam len step alpha beta1 beta2 epsilon lambda weights gradient m v =
  unsafePerformIO $ do
    let (wPtr, _) = U.unsafeToForeignPtr0 weights
    let (gPtr, _) = U.unsafeToForeignPtr0 gradient
    let (mPtr, _) = U.unsafeToForeignPtr0 m
    let (vPtr, _) = U.unsafeToForeignPtr0 v
    withForeignPtr wPtr $ \wPtr' ->
      withForeignPtr gPtr $ \gPtr' ->
        withForeignPtr mPtr $ \mPtr' ->
          withForeignPtr vPtr $ \vPtr' ->
            descend_adam_cpu len step alpha beta1 beta2 epsilon lambda wPtr' gPtr' mPtr' vPtr'
    return (U.unsafeFromForeignPtr0 wPtr len, U.unsafeFromForeignPtr0 mPtr len, U.unsafeFromForeignPtr0 vPtr len)
{-# NOINLINE descendUnsafeAdam #-}


foreign import ccall unsafe
    descend_sgd_cpu
      :: Int -> RealNum -> RealNum -> RealNum -> Ptr RealNum -> Ptr RealNum -> Ptr RealNum -> Ptr RealNum -> Ptr RealNum -> IO ()

foreign import ccall unsafe
    descend_adam_cpu
      :: Int -> Int -> RealNum -> RealNum -> RealNum -> RealNum -> RealNum -> Ptr RealNum -> Ptr RealNum -> Ptr RealNum -> Ptr RealNum -> IO ()
