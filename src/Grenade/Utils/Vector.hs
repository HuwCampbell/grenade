{-# LANGUAGE BangPatterns     #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE GADTs            #-}
{-# LANGUAGE RankNTypes       #-}
module Grenade.Utils.Vector
    ( mapVector
    , mapVectorInPlace
    , createVector
    , createVectorUnsafe
    , zipWithVector
    , zipWithVectorInPlaceSnd
    ) where


import           Control.Monad        (when)
import qualified Data.Vector.Storable as V
import           Foreign
import           GHC.ForeignPtr       (mallocPlainForeignPtrBytes)
import           GHC.Storable         (readDoubleOffPtr, writeDoubleOffPtr)
import           System.IO.Unsafe     (unsafePerformIO)


import           Grenade.Types


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
    doMalloc dummy = mallocPlainForeignPtrBytes (n * sizeOf dummy)

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
mapVectorInPlace f v =
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
zipWithVectorInPlaceSnd f u v =
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
