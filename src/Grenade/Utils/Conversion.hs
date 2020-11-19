{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE CPP                 #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeOperators       #-}

module Grenade.Utils.Conversion
  ( toLayerShape
  , toS1D
  , fromS1D
  , toS2D
  , fromS2D
  , toColumnsS2D
  , toRowsS2D
  , fromRowMajorVectorToSD1
  , fromRowMajorVectorToSD1V
  , fromRowMajorVectorToSD2
  , fromRowMajorVectorToSD2V
  ) where

import           Control.Monad
import           Data.Maybe                   (fromMaybe)
import           Data.Proxy
import           Data.Singletons
import           Data.Singletons.TypeLits
import qualified Data.Vector.Storable         as V
import qualified Data.Vector.Storable.Mutable as VM
import           Foreign
import           Foreign.C.Types
import           Foreign.Ptr
import           Foreign.Storable             (sizeOf)
import           GHC.IO.Handle.Text           (memcpy)
import           GHC.TypeLits                 hiding (natVal)
import qualified Numeric.LinearAlgebra        as NLA
import qualified Numeric.LinearAlgebra        as LA
import qualified Numeric.LinearAlgebra.Static as LAS
import           System.IO.Unsafe             (unsafePerformIO)
import           Unsafe.Coerce                (unsafeCoerce)

import           Grenade.Core.Shape
import           Grenade.Layers.Internal.BLAS
import           Grenade.Types
import           Grenade.Utils.Vector

import           Debug.Trace

toRows :: (KnownNat m, KnownNat n) => S ('D2 m n) -> [V.Vector RealNum]
toRows (S2D x)    = map LAS.extract $ LAS.toRows x
toRows x@(S2DV{}) = toRowsS2D x

toCols :: (KnownNat m, KnownNat n) => S ('D2 m n) -> [V.Vector RealNum]
toCols (S2D x)    = map LAS.extract $ LAS.toColumns x
toCols x@(S2DV{}) = toColumnsS2D x


testShapeV = fromRowMajorVectorToSD2V (V.fromList [0..9]) :: S ('D2 2 5)
testShape = fromRowMajorVectorToSD2 (V.fromList [0..9]) :: S ('D2 2 5)


-- | Converts the given vector to the correct layer shape.
toLayerShape :: S i -> S x -> S x
toLayerShape x y = case (x, y) of
  (S1D{}, S1DV{}) -> toS1D y
  (S1DV{}, S1D{}) -> fromS1D y
  (S2D{}, S2DV{}) -> toS2D y
  (S2DV{}, S2D{}) -> fromS2D y
  -- (S3D{}, S2DV{}) -> y
  -- (S3D{}, S1DV{}) -> y
  _               -> y
{-# INLINE toLayerShape #-}

toS1D :: S ('D1 l) -> S ('D1 l)
toS1D (S1DV vec) = S1D $ (fromMaybe (error $ "wrong length of vector with " ++ show (V.length vec) ++ " in toS1D ") $
                        -- test = LAS.create $ NLA.reshape 5 $ (V.fromList [0..9]) :: Maybe (LAS.L 2 5)
                         -- trace ("toS1D vec: " ++ show (vec))
                         -- trace ("toS1D inp: " ++ show (res)) res)
                         res)
                         where res = LAS.create vec
toS1D x@S1D{} = x
{-# INLINE toS1D #-}

fromS1D :: S ('D1 l) -> S ('D1 l)
fromS1D (S1D vec) = S1DV (LAS.extract vec)
fromS1D x@S1DV{}  = x
{-# INLINE fromS1D #-}

-- | Convert from vector representation.
toS2D :: forall i j . S ('D2 i j) -> S ('D2 i j)
toS2D (S2DV vec) = S2D $ LAS.matrix $ V.toList . LA.flatten . reshapeF n . LA.vector . V.toList $ vec
  where
    n = fromIntegral $ natVal (Proxy :: Proxy j)
    reshapeF r = LA.tr' . LA.reshape r

toS2D x@S2D{}    = x
{-# INLINE toS2D #-}

-- | Convert to vector representation.
fromS2D :: S ('D2 i j) -> S ('D2 i j)
fromS2D (S2D mat) = S2DV $ V.concat $ map LAS.extract $ LAS.toColumns mat
fromS2D x@S2DV{}  = x
{-# INLINE fromS2D #-}

-- test = LAS.create $ NLA.reshape 5 $ (V.fromList [0..9]) :: Maybe (LAS.L 2 5)
-- Just (matrix
--  [ 0.0, 1.0, 2.0, 3.0, 4.0
--  , 5.0, 6.0, 7.0, 8.0, 9.0 ] :: L 2 5)

-- | Efficiently extract the columns of the shape.
toColumnsS2D :: forall i j . S ('D2 i j) -> [V.Vector RealNum]
toColumnsS2D (S2D mat) = map LAS.extract . LAS.toColumns $ mat
toColumnsS2D (S2DV vec) = map (\idx -> V.slice idx m vec) [0,m .. (V.length vec - m)]
  where
    m = fromIntegral $ natVal (Proxy :: Proxy i)
{-# INLINE toColumnsS2D #-}

-- | Efficiently extract the rows of the shape.
toRowsS2D :: forall i j . S ('D2 i j) -> [V.Vector RealNum]
toRowsS2D (S2D mat) = map LAS.extract . LAS.toRows $ mat
toRowsS2D (S2DV vec) =
  unsafePerformIO $
  V.unsafeWith vec $ \from ->
    flip mapM [0 .. m - 1] $ \row -> do
      vec' <- createVector n
      V.unsafeWith vec' $ \to -> do
        let go (-1) = return ()
            go !col = do
              let idx = col * m + row
              x <- peekElemOff from idx
              pokeElemOff to col x
              go (col - 1)
        go (n - 1)
        return vec'
  where
    m = fromIntegral $ natVal (Proxy :: Proxy i)
    n = fromIntegral $ natVal (Proxy :: Proxy j)
{-# INLINE toRowsS2D #-}

fromRowMajorVectorToSD1 :: forall l . (KnownNat l) => V.Vector RealNum -> S ('D1 l)
fromRowMajorVectorToSD1 vec
  | V.length vec /= l = error $ "cannot create Vector R " ++ show l ++ " from vector with length " ++ show (V.length vec) ++ " in fromRowMajorVectorToSD1"
  | otherwise = S1D (unsafeCoerce vec)
  where l = fromIntegral $ natVal (Proxy :: Proxy l)
{-# INLINE fromRowMajorVectorToSD1 #-}

fromRowMajorVectorToSD1V :: forall l . (KnownNat l) => V.Vector RealNum -> S ('D1 l)
fromRowMajorVectorToSD1V = S1DV
{-# INLINE fromRowMajorVectorToSD1V #-}

fromRowMajorVectorToSD2 :: forall i j . (KnownNat i, KnownNat j) => V.Vector RealNum -> S ('D2 i j)
fromRowMajorVectorToSD2 vec
  | V.length vec /= m * n = error $ "cannot create matrix L " ++ show (m,n) ++ " from vector length " ++ show (V.length vec) ++ " in fromRowMajorVectorToSD2"
  | otherwise = S2D $ unsafeCoerce $ LA.reshape n vec
  where
    m = fromIntegral $ natVal (Proxy :: Proxy i)
    n = fromIntegral $ natVal (Proxy :: Proxy j)
{-# INLINE fromRowMajorVectorToSD2 #-}

fromRowMajorVectorToSD2V :: forall i j . (KnownNat i, KnownNat j) => V.Vector RealNum -> S ('D2 i j)
fromRowMajorVectorToSD2V vec = unsafePerformIO $ do
  vec' <- createVector (V.length vec)
  V.unsafeWith vec $ \from ->
    V.unsafeWith vec' $ \to ->  do
    let go (-1) = return ()
        go !k = do
          let idx = nRow * m + nCol
              nCol = k `div` n
              nRow = k `mod` n
          x <- peekElemOff from k
          pokeElemOff to idx x
          go (k-1)
    go (V.length vec - 1)
  return $ S2DV vec'

  -- S2DV $ LA.flatten . reshapeF n . LA.vector . V.toList $ vec
  where
    m = fromIntegral $ natVal (Proxy :: Proxy i)
    n = fromIntegral $ natVal (Proxy :: Proxy j)
    reshapeF r = LA.tr' . LA.reshape r
{-# INLINE fromRowMajorVectorToSD2V #-}

-- test =
--   toRowsS2D $
--   (fromRowMajorVectorToSD2V (V.fromList [0..9]) :: S ('D2 2 5))
--   -- (fromRowMajorVectorToSD1 (V.fromList [0..9]) :: S ('D1 10))
