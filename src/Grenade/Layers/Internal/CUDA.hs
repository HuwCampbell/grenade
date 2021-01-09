{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE CPP                 #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE Strict              #-}
{-# LANGUAGE TemplateHaskell     #-}
{-# LANGUAGE TypeOperators       #-}

module Grenade.Layers.Internal.CUDA
  ( setCudaTriggerSize
  , useCuda
  , cudaDescendUnsafeAdamGPU
  , cudaSumVectorsGPU
  ) where

import           Control.Concurrent
import           Control.Concurrent.Lock      (Lock)
import qualified Control.Concurrent.Lock      as Lock
import           Control.Concurrent.MVar
import           Control.Exception            (IOException, handle)
import           Control.Monad
import qualified Data.Map.Strict              as M
import           Data.Maybe                   (fromMaybe)
import           Data.Proxy
import           Data.String
import qualified Data.Vector.Storable         as V
import qualified Data.Vector.Storable         as U (unsafeFromForeignPtr0,
                                                    unsafeToForeignPtr0)
import           Foreign                      (withForeignPtr)
import           Foreign.C.Types
-- import qualified Foreign.CUDA.BLAS            as BLAS
import           Control.Monad.Primitive      (PrimBase)
import qualified Data.ByteString
import           Data.FileEmbed               (embedFile)
import qualified Foreign.CUDA.Driver          as CUDA
import           Foreign.Ptr
import           Foreign.Storable             (sizeOf)
import           GHC.IO.Handle.Text           (memcpy)
import           GHC.TypeLits
-- import qualified Numeric.LinearAlgebra        as LA
import qualified Numeric.LinearAlgebra.Static as LAS
import           System.IO.Unsafe             (unsafePerformIO)

import           Grenade.Types
import           Grenade.Utils.Vector

import           Debug.Trace


-- | Cuda context
cudaTriggerSize :: MVar (Maybe Int)
cudaTriggerSize = unsafePerformIO $ newMVar Nothing

-- | Set the minimum vector/matrix size (nr of elements) of when CUDA is used.
setCudaTriggerSize :: (PrimBase m) => Maybe Int -> m ()
setCudaTriggerSize sz = unsafePerformIO (tryPutMVar cudaTriggerSize sz) `seq` return ()

useCuda :: Int -> Bool
useCuda sz = unsafePerformIO $ do
  minSz <- tryReadMVar cudaTriggerSize
  return $ maybe False (sz >=) (join minSz)


-- Load the ptx file into the binary at compile-time!
bsGPUGradientDescent :: Data.ByteString.ByteString
bsGPUGradientDescent = $(embedFile "cbits/gpu_gradient_descent.ptx")

-- Load the ptx file into the binary at compile-time!
bsGPUSumVectors :: Data.ByteString.ByteString
bsGPUSumVectors = $(embedFile "cbits/gpu_sum_vectors.ptx")

-- | Cuda function for Adam
cudaDescendAdamGPUFun :: MVar (Maybe CUDA.Fun)
cudaDescendAdamGPUFun = unsafePerformIO $ newMVar Nothing
{-# NOINLINE cudaDescendAdamGPUFun #-}

-- | Cuda function for sumVectors
cudaSumVectorsGPUFun :: MVar (Maybe CUDA.Fun)
cudaSumVectorsGPUFun = unsafePerformIO $ newMVar Nothing
{-# NOINLINE cudaSumVectorsGPUFun #-}

-- | Cuda function for sumVectors
lock :: Lock
lock = unsafePerformIO Lock.new
{-# NOINLINE lock #-}


writeMVar :: MVar a -> a -> IO ()
writeMVar var x = modifyMVar var (\_ -> return (x, ()))

-- cudaInitialise :: IO ()
-- cudaInitialise = do
--   CUDA.initialise []
--   count <- CUDA.count
--   devs <- mapM CUDA.device [0 .. count - 1]
--   putStrLn $ "Initialising CUDA. Found " ++ show count ++ " devices: "
--   mapM_ (CUDA.props >=> print) devs
--   putStrLn ""
--   unless (null devs) $ do
--     props <- CUDA.props (head devs)
--     let threadsPerBlock = CUDA.maxThreadsPerBlock props
--     putStrLn $ "Using CUDA with device: " ++ show (CUDA.deviceName props) ++ "; threads per Block: " ++ show threadsPerBlock ++ "\n"
--     writeIORef cudaThreadsPerBlock threadsPerBlock
--     device <- CUDA.device 0
--     writeIORef cudaDevice (Just device)
--     ctx <- CUDA.create device []
--     writeIORef cudaCtx (Just ctx)
--     -- Compile to ptx?
--     mdl <- CUDA.loadData bsGPUGradientDescent
--     descendAdamGpuFun <- CUDA.getFun mdl (fromString "descend_adam_gpu")
--     writeIORef cudaDescendAdamGPUFun (Just descendAdamGpuFun)

cudaInitialise :: IO ()
cudaInitialise = do
  threadId <- myThreadId
  Lock.acquire lock
  -- m <- readIORef cudaCtx
  -- case M.lookup threadId m of
  --   Just{} -> return ()
  --   Nothing -> do
  do
      CUDA.initialise []
      count <- CUDA.count
      devs <- mapM CUDA.device [0 .. count - 1]
      if null devs
        then return ()
        else do
          putStrLn $ "Initialising CUDA. Found " ++ show count ++ " devices: "
          mapM_ (CUDA.props >=> print) devs
          putStrLn ""
          props <- CUDA.props (head devs)
          let threadsPerBlock = CUDA.maxThreadsPerBlock props
          putStrLn $ "Using CUDA with device: " ++ show (CUDA.deviceName props) ++ "; threads per Block: " ++ show threadsPerBlock ++ "\n"
          writeMVar cudaThreadsPerBlock threadsPerBlock
          device <- CUDA.device 0
          writeMVar cudaDevice (Just device)
          -- ctx <- CUDA.create device []
          -- -- atomicModifyIORef cudaCtx (\m' -> (M.insert threadId ctx m', ()))
          -- -- writeIORef cudaCtx (M.Just ctx)
          --      -- Inline compile to ptx?
          -- flip mapM_ functions $ \(file, functionName, ioRef) -> do
          --   mdl <- CUDA.loadData file
          --   fun <- CUDA.getFun mdl (fromString functionName)
          --   modifyMVar ioRef (\_ -> return (Just fun, ()))
  Lock.release lock
  where
    functions =
      [ (bsGPUGradientDescent, "descend_adam_gpu", cudaDescendAdamGPUFun) -- ^ Adam
      , (bsGPUSumVectors, "sum_vectors_gpu", cudaSumVectorsGPUFun) -- ^ Sum a list of vectors
      ]


-- | Cuda context
cudaThreadsPerBlock :: MVar Int
cudaThreadsPerBlock = unsafePerformIO $ newMVar 128


-- | Cuda device
cudaDevice :: MVar (Maybe CUDA.Device)
cudaDevice = unsafePerformIO $ newMVar Nothing

withCudaCtx :: CUDA.Device -> (CUDA.Fun -> IO b) -> IO b
withCudaCtx device f = do
  -- device <- fromMaybe (error "empty device") . join <$> tryReadMVar cudaDevice -- device 0
  -- myThreadId >>= print
  !ctx <- CUDA.create device []
  mdl <- CUDA.loadData bsGPUSumVectors
  fun <- CUDA.getFun mdl (fromString "sum_vectors_gpu")
  !b <- f fun
  CUDA.destroy ctx
  return b

-- | To inform the user when it works again
cudaErrored :: MVar Bool
cudaErrored = unsafePerformIO $ newMVar False
{-# NOINLINE cudaErrored #-}


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
    mFun <- join <$> tryReadMVar cudaDescendAdamGPUFun
    case mFun of
      Nothing -> cudaInitialise >> tryReadMVar cudaDescendAdamGPUFun >>= maybe (return Nothing) cudaDescendAdamGPU' -- Ensure it was initialised
      x -> handle handler (cudaDescendAdamGPU' x)
  where
    handler (e :: CUDA.CUDAException) = writeMVar cudaErrored True >> print e >> return Nothing
    cudaDescendAdamGPU' mFun =
      case mFun of
        Nothing -> return Nothing
        Just fun -> do
          threadsPerBlock <- fromMaybe 128 <$> tryReadMVar cudaThreadsPerBlock
          V.unsafeWith weights $ \wPtr' ->
            V.unsafeWith gradient $ \gPtr' ->
              V.unsafeWith m $ \mPtr' ->
                V.unsafeWith v $ \vPtr' -> do
                  wPtrDev <- CUDA.mallocArray len :: IO (CUDA.DevicePtr RealNum)
                  gPtrDev <- CUDA.mallocArray len :: IO (CUDA.DevicePtr RealNum)
                  mPtrDev <- CUDA.mallocArray len :: IO (CUDA.DevicePtr RealNum)
                  vPtrDev <- CUDA.mallocArray len :: IO (CUDA.DevicePtr RealNum)
                  CUDA.pokeArray len wPtr' wPtrDev
                  CUDA.pokeArray len gPtr' gPtrDev
                  CUDA.pokeArray len mPtr' mPtrDev
                  CUDA.pokeArray len vPtr' vPtrDev
                  let tx = ceiling (fromIntegral len / (fromIntegral threadsPerBlock :: Float)) :: Int
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
                    , CUDA.VArg wPtrDev
                    , CUDA.VArg gPtrDev
                    , CUDA.VArg mPtrDev
                    , CUDA.VArg vPtrDev
                    ]
                  CUDA.sync
                  CUDA.peekArray len wPtrDev wPtr'
                  CUDA.peekArray len mPtrDev mPtr'
                  CUDA.peekArray len vPtrDev vPtr'
                  mapM_ CUDA.free [wPtrDev,gPtrDev,mPtrDev,vPtrDev] -- put pointers back for reuse
                  err <- fromMaybe False <$> tryReadMVar cudaErrored
                  when err $ writeMVar cudaErrored False >> putStrLn "Cuda works again"
                  return $ Just (weights, m, v)
{-# NOINLINE cudaDescendUnsafeAdamGPU #-}

cudaSumVectorsGPU :: [V.Vector RealNum] -> Maybe (V.Vector RealNum) -- vectors to sum, all of same length
cudaSumVectorsGPU [] = Nothing
cudaSumVectorsGPU vs
  | useCuda (length vs * V.length (head vs)) =
    unsafePerformIO $! do
      mDev <- join <$> tryReadMVar cudaDevice
      case mDev of
        Nothing -> cudaInitialise >> tryReadMVar cudaDevice >>= maybe (return Nothing) cudaSumVectorsGPU' -- Ensure it was initialised
        dev -> handle handler (cudaSumVectorsGPU' dev)
  | otherwise = Nothing
  where
    handler (e :: CUDA.CUDAException) = writeMVar cudaErrored True >> print e >> return Nothing
    cudaSumVectorsGPU' (mDev :: Maybe CUDA.Device) =
      case mDev of
        Nothing -> return Nothing
        Just dev ->
          withCudaCtx dev $ \fun -> do
            let outVec = createVectorUnsafe (V.length $ head vs)
            let outLen = V.length (head vs)
                inLen = outLen * length vs
            let inVec = V.generate inLen genFun
                genFun idx =
                  let vIdx = idx `div` outLen
                      vec = vs !! vIdx
                   in vec V.! (idx - vIdx * outLen)
            threadsPerBlock <- fromMaybe 128 <$> tryReadMVar cudaThreadsPerBlock
            V.unsafeWith inVec $ \inVecPtr' -> do
              V.unsafeWith outVec $ \outVecPtr' -> do
                inVecPtrDev <- CUDA.mallocArray inLen :: IO (CUDA.DevicePtr RealNum)
                outVecPtrDev <- CUDA.mallocArray outLen :: IO (CUDA.DevicePtr RealNum)
                CUDA.pokeArray inLen inVecPtr' inVecPtrDev
                let tx = ceiling (fromIntegral outLen / (fromIntegral threadsPerBlock :: Float)) :: Int
                CUDA.launchKernel
                  fun
                  (tx, 1, 1 :: Int)
                  (threadsPerBlock, 1, 1)
                  0
                  Nothing
                  [ CUDA.IArg (fromIntegral inLen)
                  , CUDA.IArg (fromIntegral outLen)
                  , CUDA.VArg (inVecPtrDev :: CUDA.DevicePtr RealNum)
                  , CUDA.VArg (outVecPtrDev :: CUDA.DevicePtr RealNum)
                  ]
                CUDA.sync
                CUDA.peekArray outLen outVecPtrDev outVecPtr'
                mapM_ CUDA.free [inVecPtrDev, outVecPtrDev] -- put pointers back for reuse
                err <- fromMaybe False <$> tryReadMVar cudaErrored
                when err $ writeMVar cudaErrored False >> putStrLn "Cuda works again"
                return $ Just outVec
{-# NOINLINE cudaSumVectorsGPU #-}


-- test =
--   cudaSumVectorsGPU (map V.fromList (replicate 32 [1..10]))
