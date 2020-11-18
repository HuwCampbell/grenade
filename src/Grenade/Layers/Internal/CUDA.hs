{-# LANGUAGE CPP                 #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE Strict              #-}
{-# LANGUAGE TypeOperators       #-}

module Grenade.Layers.Internal.CUDA
    ( -- easy interface
      getCudaCtx
    , cudaDestroy -- if you use CUDA, destroy the context after the ANN is not needed anymore!
    , cudaDescendUnsafeAdamGPU
    ) where

import           Control.Monad
import           Data.IORef
import qualified Data.Map.Strict              as M
import           Data.Proxy
import           Data.String
import qualified Data.Vector.Storable         as V
import qualified Data.Vector.Storable         as U (unsafeFromForeignPtr0,
                                                    unsafeToForeignPtr0)
import           Foreign                      (withForeignPtr)
import           Foreign.C.Types
-- import qualified Foreign.CUDA.BLAS            as BLAS
import qualified Foreign.CUDA.Driver          as CUDA
import           Foreign.Ptr
import           Foreign.Storable             (sizeOf)
import           GHC.IO.Handle.Text           (memcpy)
import           GHC.TypeLits
import qualified Numeric.LinearAlgebra        as LA
import qualified Numeric.LinearAlgebra.Static as LAS
import           System.IO.Unsafe             (unsafePerformIO)

import           Grenade.Types

import           Debug.Trace

-- DGEMM seems to be a little faster, and CBLAS has some overhead. So we aim for implementing all operations with DGEMM and use the direct calls to BLAS

#define USE_DGEMM_ONLY 0
#define USE_CBLAS 0

cudaInitialise :: IO ()
cudaInitialise = do
  CUDA.initialise []
  count <- CUDA.count
  devs <- mapM CUDA.device [0 .. count - 1]
  putStrLn $ "Initialising CUDA. Found " ++ show count ++ " devices: "
  mapM_ (CUDA.props >=> print) devs
  putStrLn ""
  unless (null devs) $ do
    props <- CUDA.props (head devs)
    putStrLn $ "Using CUDA with device: " ++ show (CUDA.deviceName props) ++ "\n"
    device <- CUDA.device 0
    writeIORef cudaDevice (Just device)
    ctx <- CUDA.create device []
    writeIORef cudaCtx (Just ctx)
    -- Compile to ptx?
    mdl <- CUDA.loadFile "cbits/gpu_gradient_descent.ptx"
    descendAdamGpuFun <- CUDA.getFun mdl (fromString "descend_adam_gpu")
    writeIORef cudaDescendAdamGPUFun (Just descendAdamGpuFun)


-- | If you use CUDA, destroy the context after the ANN is not needed anymore!
cudaDestroy :: IO ()
cudaDestroy = do
  mCtx <- readIORef cudaCtx
  maybe (return ()) CUDA.destroy mCtx

-- | Cuda device
cudaDevice :: IORef (Maybe CUDA.Device)
cudaDevice = unsafePerformIO $ newIORef Nothing

-- | Cuda context
cudaCtx :: IORef (Maybe CUDA.Context)
cudaCtx = unsafePerformIO $ newIORef Nothing

getCudaCtx :: IO (Maybe CUDA.Context)
getCudaCtx = do
  mCtx <- readIORef cudaCtx
  case mCtx of
    Just{}  -> return mCtx
    Nothing -> cudaInitialise >> readIORef cudaCtx

-- | Cuda function for Adam
cudaDescendAdamGPUFun :: IORef (Maybe CUDA.Fun)
cudaDescendAdamGPUFun = unsafePerformIO $ newIORef Nothing

devicePtrs :: IORef (M.Map Int [CUDA.DevicePtr RealNum])
devicePtrs = unsafePerformIO $ newIORef mempty
{-# NOINLINE devicePtrs #-}

-- | Get a device pointer.
getDevicePtr :: Int -> IO (CUDA.DevicePtr RealNum)
getDevicePtr size = do
  mPtr <- atomicModifyIORef devicePtrs $ \m ->
    case M.lookup size m of
      Nothing     -> (m, Nothing)
      Just []     -> (m, Nothing)
      Just (x:xs) -> (M.insert size xs m, Just x)
  maybe (CUDA.mallocArray size) return mPtr

-- | Return a device pointer.
putDevicePtr :: Int -> CUDA.DevicePtr RealNum -> IO ()
putDevicePtr size ptr = atomicModifyIORef devicePtrs $ \m -> (M.insertWith (++) size [ptr] m, ())


cudaDescendUnsafeAdamGPU ::
     Int -- Len
  -> Int -- Step
  -> RealNum -- Alpha
  -> RealNum -- Beta1
  -> RealNum -- Beta2
  -> RealNum -- Epsilon
  -> RealNum -- Lambda
  -> V.Vector RealNum -- Weights
  -> V.Vector RealNum -- Gradient
  -> V.Vector RealNum -- M
  -> V.Vector RealNum -- V
  -> Maybe (V.Vector RealNum, V.Vector RealNum, V.Vector RealNum)
cudaDescendUnsafeAdamGPU len step alpha beta1 beta2 epsilon lambda weights gradient m v =
  unsafePerformIO $ do
    mFun <- readIORef cudaDescendAdamGPUFun
    case mFun of
      Nothing -> cudaInitialise >> readIORef cudaDescendAdamGPUFun >>= cudaDescendAdamGPU' -- Ensure it was initialised
      x -> cudaDescendAdamGPU' x
  where
    cudaDescendAdamGPU' mFun = do
      case mFun of
        Nothing -> return Nothing
        Just fun -> do
          let (wPtr, _) = U.unsafeToForeignPtr0 weights
          let (gPtr, _) = U.unsafeToForeignPtr0 gradient
          let (mPtr, _) = U.unsafeToForeignPtr0 m
          let (vPtr, _) = U.unsafeToForeignPtr0 v
          withForeignPtr wPtr $ \wPtr' ->
            withForeignPtr gPtr $ \gPtr' ->
              withForeignPtr mPtr $ \mPtr' ->
                withForeignPtr vPtr $ \vPtr' -> do
                  wPtrDev <- getDevicePtr len
                  gPtrDev <- getDevicePtr len :: IO (CUDA.DevicePtr RealNum)
                  mPtrDev <- getDevicePtr len :: IO (CUDA.DevicePtr RealNum)
                  vPtrDev <- getDevicePtr len :: IO (CUDA.DevicePtr RealNum)
                  CUDA.pokeArray len wPtr' wPtrDev
                  CUDA.pokeArray len gPtr' gPtrDev
                  CUDA.pokeArray len mPtr' mPtrDev
                  CUDA.pokeArray len vPtr' vPtrDev
                  let tx = ceiling (fromIntegral len / threadsPerBlockDbl) :: Int
                  -- print (len, step+1)
                  -- print (alpha, beta1, beta2, epsilon, lambda)

                  CUDA.launchKernel
                    fun
                    (tx, 1, 1 :: Int)
                    (threadsPerBlock, 1, 1)
                    0
                    Nothing
                    [ CUDA.IArg (fromIntegral len)
                    , CUDA.IArg (fromIntegral $ step + 1)
                    , CUDA.VArg (alpha :: RealNum)
                    , CUDA.VArg (beta1 :: RealNum)
                    , CUDA.VArg (beta2 :: RealNum)
                    , CUDA.VArg (epsilon :: RealNum)
                    , CUDA.VArg (lambda :: RealNum)
                    -- , CUDA.FArg (realToFrac alpha :: Float)
                    -- , CUDA.FArg (realToFrac beta1 :: Float)
                    -- , CUDA.FArg (realToFrac beta2 :: Float)
                    -- , CUDA.FArg (realToFrac epsilon :: Float)
                    -- , CUDA.FArg (realToFrac lambda :: Float)
                    , CUDA.VArg wPtrDev
                    , CUDA.VArg gPtrDev
                    , CUDA.VArg mPtrDev
                    , CUDA.VArg vPtrDev
                    ]
                  CUDA.sync
                  CUDA.peekArray len wPtrDev wPtr'
                  CUDA.peekArray len mPtrDev mPtr'
                  CUDA.peekArray len vPtrDev vPtr'
                  putDevicePtr len wPtrDev -- put pointers back for reuse
                  putDevicePtr len gPtrDev
                  putDevicePtr len mPtrDev
                  putDevicePtr len vPtrDev
                  -- putStrLn "Freed GPU"
                  -- descend_adam_cpu len step alpha beta1 beta2 epsilon lambda wPtr' gPtr' mPtr' vPtr'
          return $ Just (U.unsafeFromForeignPtr0 wPtr len, U.unsafeFromForeignPtr0 mPtr len, U.unsafeFromForeignPtr0 vPtr len)
    threadsPerBlock = 32 :: Int
    threadsPerBlockDbl = fromIntegral threadsPerBlock :: Double
{-# NOINLINE cudaDescendUnsafeAdamGPU #-}
