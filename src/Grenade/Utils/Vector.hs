{-# LANGUAGE BangPatterns     #-}
{-# LANGUAGE CPP              #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE GADTs            #-}
{-# LANGUAGE RankNTypes       #-}
{-# LANGUAGE Strict           #-}
module Grenade.Utils.Vector
    ( unsafeMemCopyVectorFromTo
    , memCopyVectorFromTo
    , unsafeMemZero
    , memZero
    , mapVector
    , mapVectorInPlace
    , createVector
    , createVectorUnsafe
    , zipWithVector
    , zipWithVectorInPlaceSnd
    , sumVectors
    ) where


import           Control.Monad        (when)
import qualified Data.Vector          as VB
import qualified Data.Vector.Storable as V
import           Foreign
import           Foreign.C.Types
import           GHC.ForeignPtr       (mallocPlainForeignPtrAlignedBytes,
                                       mallocPlainForeignPtrBytes)
import           GHC.IO.Handle.Text   (memcpy)
import           System.IO.Unsafe     (unsafePerformIO)


import           Grenade.Types


-- | Memory copy a vector from one to the other.
unsafeMemCopyVectorFromTo :: V.Vector RealNum -> V.Vector RealNum -> V.Vector RealNum
unsafeMemCopyVectorFromTo from to = unsafePerformIO (memCopyVectorFromTo from to)
{-# NOINLINE unsafeMemCopyVectorFromTo #-}

-- | Memory copy a vector from one to the other.
memCopyVectorFromTo :: V.Vector RealNum -> V.Vector RealNum -> IO (V.Vector RealNum)
memCopyVectorFromTo from to = do
  V.unsafeWith from $ \fromPtr' ->
    V.unsafeWith to $ \toPtr' ->
      void $ memcpy toPtr' fromPtr' (fromIntegral $ sizeOf (V.head from) * V.length to)
  return to
{-# INLINE memCopyVectorFromTo #-}


-- | Write zero to all elements in a vector.
unsafeMemZero :: V.Vector RealNum -> V.Vector RealNum
unsafeMemZero = unsafePerformIO . memZero


-- | Write zero to all elements in a vector.
memZero :: V.Vector RealNum -> IO (V.Vector RealNum)
memZero vec = do
  V.unsafeWith vec $ \vecPtr' ->
    void $ memset vecPtr' 0 (fromIntegral $ sizeOf (0 :: RealNum) * V.length vec)
  return vec
{-# INLINE memZero #-}

foreign import ccall unsafe "string.h" memset  :: Ptr a -> CInt  -> CSize -> IO (Ptr a)


-- | allocates memory for a new vector (code from HMatrix)
createVectorUnsafe :: V.Storable a => Int -> V.Vector a
createVectorUnsafe = unsafePerformIO . createVector
{-# NOINLINE createVectorUnsafe #-}


-- | allocates memory for a new vector (code from HMatrix)
createVector :: V.Storable a => Int -> IO (V.Vector a)
createVector n = do
  when (n < 0) $ error ("trying to createVector of negative dim: " ++ show n)
  fp <- doMalloc undefined
  return $ V.unsafeFromForeignPtr fp 0 n
    --
    -- Use the much cheaper Haskell heap allocated storage
    -- for foreign pointer space we control
    --
  where
    doMalloc :: V.Storable b => b -> IO (ForeignPtr b)
    doMalloc ~dummy =
      -- memory aligned operations are faster (we use double, so align to 64)
      -- ref: Vanhoucke, Vincent, Andrew Senior, and Mark Z. Mao. "Improving the speed of neural networks on CPUs." (2011).
      -- non-aligned:
      -- mallocPlainForeignPtrBytes (n * sizeOf dummy)
-- #ifdef USE_DOUBLE
--       mallocPlainForeignPtrAlignedBytes (n * sizeOf dummy) 64
-- #else
      mallocPlainForeignPtrAlignedBytes (n * sizeOf dummy) 16
-- #endif

-- | map on Vectors (code from HMatrix)
mapVector :: (V.Storable a, V.Storable b) => (a -> b) -> V.Vector a -> V.Vector b
mapVector f v =
  unsafePerformIO $ do
    w <- createVector (V.length v)
    V.unsafeWith v $ \p ->
      V.unsafeWith w $ \q -> do
        let go (-1) = return ()
            go !k = do
              x <- peekElemOff p k
              pokeElemOff q k (f x)
              go (k - 1)
        go (V.length v - 1)
    return w
{-# INLINE mapVector #-}

-- | map on Vectors (code from HMatrix)
mapVectorInPlace :: (RealNum -> RealNum) -> V.Vector RealNum -> V.Vector RealNum
mapVectorInPlace f v = -- mapVector f v
  unsafePerformIO $ do
    V.unsafeWith v $ \p -> do
        let go (-1) = return ()
            go !k = do
              x <- peekElemOff p k
              pokeElemOff p k (f x)
              go (k - 1)
        go (V.length v - 1)
    return v
{-# INLINE mapVectorInPlace #-}


-- | zipWith for Vectors (code from HMatrix)
zipWithVector :: (RealNum -> RealNum -> RealNum) -> V.Vector RealNum -> V.Vector RealNum -> V.Vector RealNum
zipWithVector f u v =
  unsafePerformIO $ do
    let n = min (V.length u) (V.length v)
    w <- createVector n
    V.unsafeWith u $ \pu ->
      V.unsafeWith v $ \pv ->
        V.unsafeWith w $ \pw -> do
          let go (-1) = return ()
              go !k = do
                x <- peekElemOff pu k
                y <- peekElemOff pv k
                pokeElemOff pw k (f x y)
                go (k - 1)
          go (n - 1)
    return w
{-# INLINE zipWithVector #-}


-- | zipWith two vectors and replace the second with the result
zipWithVectorInPlaceSnd :: (RealNum -> RealNum -> RealNum) -> V.Vector RealNum -> V.Vector RealNum -> V.Vector RealNum
zipWithVectorInPlaceSnd f u v = -- zipWithVector f u v
  unsafePerformIO $ do
    let n = min (V.length u) (V.length v)
    V.unsafeWith u $ \pu ->
      V.unsafeWith v $ \pv -> do
        let go (-1) = return ()
            go !k = do
              x <- peekElemOff pu k
              y <- peekElemOff pv k
              pokeElemOff pv k (f x y)
              go (k - 1)
        go (n - 1)
    return v
{-# INLINE zipWithVectorInPlaceSnd #-}


-- | zipWith for Vectors (code from HMatrix)
sumVectors :: [V.Vector RealNum] -> V.Vector RealNum
sumVectors [] = V.empty
sumVectors vs =
  -- unsafePerformIO $ do
  --   let len = V.length (head vs)
  --   out <- memCopyVectorFromTo (head vs) (createVectorUnsafe len)
  --   V.unsafeWith out $ \pout -> do
  --     let go (-1) _ = return ()
  --         go idx v = do
  --           x <- peekElemOff v idx
  --           t <- peekElemOff pout idx
  --           pokeElemOff pout idx (x + t)
  --           go (idx - 1) v
  --     mapM_ (\v -> V.unsafeWith v (go (len - 1))) (tail vs)
  --   return out
  unsafePerformIO $ do
    let len = V.length (head vs)
    out <- memZero (createVectorUnsafe len)
    V.unsafeWith out $ \pout -> do
      let w = foldl (\acc v -> V.unsafeWith v : acc) [] vs
      let go (-1) _ = return ()
          go idx v = do
            x <- peekElemOff v idx
            t <- peekElemOff pout idx
            pokeElemOff pout idx (x + t)
            go (idx - 1) v
      mapM_ (\f -> f (go (len - 1))) w
    return out
{-# INLINE sumVectors #-}

foreign import ccall unsafe "sum_vectors" dger_direct :: CInt -> CInt -> Ptr (Ptr Double) -> Ptr Double -> Ptr Double
