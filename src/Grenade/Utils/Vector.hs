{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE GADTs            #-}
{-# LANGUAGE RankNTypes       #-}
module Grenade.Utils.Vector
    ( parMapVector
    , parZipWithVector
    -- Better performance when using C directly
    , parMapVectorC
    , parZipWithVectorReplSndC
      -- Functions implemented in C
    , c_times
      -- Activations implemented in C
    , c_relu
    , c_relu_dif
    , c_relu_dif_fast
    , c_leaky_relu
    , c_leaky_relu_dif
    , c_leaky_relu_dif_fast
    , c_sigmoid
    , c_sigmoid_dif
    , c_sigmoid_dif_fast
    ) where


import           Control.Concurrent.MVar
import qualified Data.Map.Strict              as M

import           Control.DeepSeq              (force)
import           Control.Monad                (zipWithM_)
import           Control.Monad.Primitive      (PrimMonad)
import           Control.Monad.ST
import           Control.Parallel.Strategies
import qualified Data.Vector.Storable         as V
import qualified Data.Vector.Storable         as U (unsafeFromForeignPtr0,
                                                    unsafeToForeignPtr0)
import qualified Data.Vector.Storable         as V
import qualified Data.Vector.Storable.Mutable as VM
import           Foreign                      (mallocForeignPtrArray, withForeignPtr)
import           Foreign.C.Types
import           Foreign.Ptr
import           GHC.Conc                     (numCapabilities)
import           GHC.Storable                 (readDoubleOffPtr, writeDoubleOffPtr)
import           System.IO.Unsafe             (unsafePerformIO)

import           Grenade.Types

import           Debug.Trace

-- ^ Usually we have 2 threads per CPU and we aim for a little less than that value.
effectiveCPUs :: Int
effectiveCPUs = max 1 $ (numCapabilities `div` 2) - 1

minChunkSize :: Int
minChunkSize = 100

-- | This function replaces the elements in the vector. Thus use with care!
parMapVector :: (RealNum -> RealNum) -> V.Vector RealNum -> V.Vector RealNum
parMapVector f vec = force (sliced vec) `seq` vec
  where
    vLen = V.length vec
    chunkSize = max minChunkSize $ vLen `div` effectiveCPUs
    sliced :: V.Vector RealNum -> [()]
    sliced vec' = map (\(start, len) -> unsafeMapVector f (V.slice start len vec') `using` rpar) splits `using` rdeepseq -- use rdeepseq to ensure the whole list is evaluated!
      where
       -- This function is called often and thus creating and destroying this data structure over and over again is costly
       splits = cached vLen $ map (\s -> (s, min (vLen - s) chunkSize)) [0,chunkSize .. (vLen - 1)]

-- | This function replaces the elements in the vector. Thus use with care!
parMapVectorC :: FunPtr (RealNum -> RealNum) -> V.Vector RealNum -> V.Vector RealNum
parMapVectorC f vec = force (sliced vec) `seq` vec
  where
    vLen = V.length vec
    chunkSize = max minChunkSize $ vLen `div` effectiveCPUs
    sliced :: V.Vector RealNum -> [()]
    sliced vec'
      | vLen <= minChunkSize = [unsafeMapVectorC f vec'] `using` rdeepseq
      | otherwise = parMap rdeepseq (\(start, len) -> unsafeMapVectorC f (V.slice start len vec')) splits -- use rdeepseq to ensure the whole list is evaluated!
        -- This function is called often and thus creating and destroying this data structure over and over again is costly
      where
        splits = cached vLen $ map (\s -> (s, min (vLen - s) chunkSize)) [0,chunkSize .. (vLen - 1)]

-- | Caching of data
type CacheKey = Int
type CacheVal = [(Int, Int)]

cacheMVar :: MVar (M.Map CacheKey CacheVal)
cacheMVar = unsafePerformIO $ newMVar mempty
{-# NOINLINE cacheMVar #-}

addCache :: CacheKey -> CacheVal -> IO ()
addCache k val = modifyMVar_ cacheMVar (return . M.insert k val)

lookupCache :: CacheKey -> IO (Maybe CacheVal)
lookupCache k = (M.lookup k =<<) <$> tryReadMVar cacheMVar


-- | Get output of function f, if possible from cache according to key (st).
cached :: CacheKey -> CacheVal -> CacheVal
cached st ~val = unsafePerformIO $ do
  c <- lookupCache st
  case c of
    Nothing -> do
      addCache st val
      return val
    Just res -> return res

{-# NOINLINE unsafeMapVector #-}
unsafeMapVector :: (RealNum -> RealNum) -> V.Vector RealNum -> ()
unsafeMapVector f vec =
  unsafePerformIO $ do
    let (vPtr, _) = U.unsafeToForeignPtr0 vec
    withForeignPtr vPtr $ \vPtr' ->
      mapM_ (\idx -> writeDoubleOffPtr vPtr' idx (f $ vec V.! idx) ) idxs
  where
    idxs = [0 .. V.length vec - 1]

{-# NOINLINE unsafeMapVectorC #-}
unsafeMapVectorC :: FunPtr (RealNum -> RealNum) -> V.Vector RealNum -> ()
unsafeMapVectorC f vec =
  unsafePerformIO $ do
    let (vPtr, _) = U.unsafeToForeignPtr0 vec
    withForeignPtr vPtr $ \vPtr' -> map_vector f vPtr' (V.length vec)


parZipWithVector :: (RealNum -> RealNum -> RealNum) -> V.Vector RealNum -> V.Vector RealNum -> V.Vector RealNum
parZipWithVector f vec1 vec2 = V.modify sliced shorter
  where
    (shorter, longer)
      | V.length vec1 <= V.length vec2 = (vec1, vec2)
      | otherwise = (vec2, vec1)
    sliced :: forall s. V.MVector s RealNum -> ST s ()
    sliced v1 = sliced' v1 longer
    chunkSize = max minChunkSize $ V.length shorter `div` effectiveCPUs
    sliced' :: forall s. V.MVector s RealNum -> V.Vector RealNum -> ST s ()
    sliced' v1 v2
      | len <= chunkSize = mapM_ (\i -> VM.modify v1 (\val1 -> f val1 (v2 V.! i)) i) [0 .. VM.length v1 - 1]
      | otherwise = do
        sliced' v1First v2First `using` rpar
        sliced' v1Second v2Second `using` rpar
      where
        (v1First, v1Second) = VM.splitAt idx v1 -- this uses unsafeSlice, e.g. does not create a new vector
        (v2First, v2Second) = V.splitAt idx v2
        idx = len `div` 2
        len = VM.length v1


-- | This function replaces the second vector
parZipWithVectorReplSndC :: FunPtr (RealNum -> RealNum -> RealNum) -> V.Vector RealNum -> V.Vector RealNum -> V.Vector RealNum
parZipWithVectorReplSndC f vec1 vec2
  | vLen /= V.length vec2 = error $ "parZipWithVectorC expected both vectors having the same length: " ++ show (V.length vec1) ++ " /= " ++ show (V.length vec2)
  | otherwise = force (sliced vec1 vec2) `seq` vec2
  where
    vLen = V.length vec1
    chunkSize = max minChunkSize $ vLen `div` effectiveCPUs
    sliced :: V.Vector RealNum -> V.Vector RealNum -> [()]
    sliced vec1' vec2'
      | vLen <= minChunkSize = [unsafeZipWithVectorReplSndC f vec1' vec2'] `using` rdeepseq
      | otherwise = parMap rdeepseq (\(start, len) -> unsafeZipWithVectorReplSndC f (V.slice start len vec1') (V.slice start len vec2')) splits -- use rdeepseq to ensure the whole list is evaluated!
      where
        -- This function is called often and thus creating and destroying this data structure over and over again is costly
        splits = cached vLen $ map (\s -> (s, min (vLen - s) chunkSize)) [0,chunkSize .. (vLen - 1)]

unsafeZipWithVectorReplSndC :: FunPtr (RealNum -> RealNum -> RealNum) -> V.Vector RealNum -> V.Vector RealNum -> ()
unsafeZipWithVectorReplSndC f vec1 vec2 =
  unsafePerformIO $ do
    let (v1Ptr, _) = U.unsafeToForeignPtr0 vec1
    let (v2Ptr, _) = U.unsafeToForeignPtr0 vec2
    withForeignPtr v1Ptr $ \v1Ptr' ->
      withForeignPtr v2Ptr $ \v2Ptr' ->
      zip_with_vector_repl_snd f v1Ptr' v2Ptr' (V.length vec1)


-- FFI calls (Foreign function calls)

foreign import ccall "map_vector"
    map_vector :: FunPtr (RealNum -> RealNum) -> Ptr RealNum -> Int -> IO ()

foreign import ccall "zip_with_vector_repl_snd"
    zip_with_vector_repl_snd :: FunPtr (RealNum -> RealNum -> RealNum) -> Ptr RealNum -> Ptr RealNum -> Int -> IO ()


-- Times

foreign import ccall "&c_times"
  c_times :: FunPtr (RealNum -> RealNum -> RealNum) -- x * y


-- Relu

foreign import ccall "&c_relu"
  c_relu :: FunPtr (RealNum -> RealNum)

-- foreign import ccall "&c_relu_zip"
--   c_relu_zip :: FunPtr (RealNum -> RealNum -> RealNum)


foreign import ccall "&c_relu_dif"
  c_relu_dif :: FunPtr (RealNum -> RealNum)

foreign import ccall "&c_relu_dif_fast"
  c_relu_dif_fast :: FunPtr (RealNum -> RealNum -> RealNum)


-- Leaky_Relu

foreign import ccall "&c_leaky_relu"
  c_leaky_relu :: FunPtr (RealNum -> RealNum)

foreign import ccall "&c_leaky_relu_dif"
  c_leaky_relu_dif :: FunPtr (RealNum -> RealNum)

foreign import ccall "&c_leaky_relu_dif_fast"
  c_leaky_relu_dif_fast :: FunPtr (RealNum -> RealNum -> RealNum)

-- Sigmoid

foreign import ccall "&c_sigmoid"
  c_sigmoid :: FunPtr (RealNum -> RealNum)

foreign import ccall "&c_sigmoid_dif"
  c_sigmoid_dif :: FunPtr (RealNum -> RealNum)

foreign import ccall "&c_sigmoid_dif_fast"
  c_sigmoid_dif_fast :: FunPtr (RealNum -> RealNum -> RealNum)
