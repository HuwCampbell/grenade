{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE CPP                 #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeOperators       #-}

module Grenade.Utils.Conversion
  ( -- Column major mode helper functions
    NrColumns
  , Row
  , Col
  , ColumnMajorModeIndex
  , columnMajorModeIndex
  , columnMajorModeRowCol

  , toLayerShape
  -- 1D conversions
  , toS1D
  , toS1DV
  , fromS1D
  , fromS1DV
  -- 2D converions
  , toS2D
  , toS2DV
  , fromS2D
  , fromS2DV
  -- further conversions
  , toRows
  , toCols
  , toColumnsS2D
  , toRowsS2D
  , fromRowMajorVectorToSD1
  , fromRowMajorVectorToSD1V
  , fromRowMajorVectorToSD2
  , fromRowMajorVectorToSD2V
  , fromColumnMajorVectorToSD2V
  ) where

import           Data.Maybe                   (fromJust, fromMaybe)
import           Data.Proxy
import           Data.Singletons.TypeLits
import qualified Data.Vector.Storable         as V
import           Foreign
import qualified Numeric.LinearAlgebra        as LA
import qualified Numeric.LinearAlgebra.Static as LAS
import           System.IO.Unsafe             (unsafePerformIO)
import           Unsafe.Coerce                (unsafeCoerce)

import           Grenade.Core.Shape
import           Grenade.Types
import           Grenade.Utils.Vector


-- Column major mode helper functions

type NrColumns = Int
type Row = Int
type Col = Int
type ColumnMajorModeIndex = Int


columnMajorModeIndex :: NrColumns -> Row -> Col -> ColumnMajorModeIndex
columnMajorModeIndex cols row col = row * cols + col

columnMajorModeRowCol :: NrColumns -> ColumnMajorModeIndex -> (Row, Col)
columnMajorModeRowCol cols idx = (idx `div` cols, idx `mod` cols)


-- import           Debug.Trace

-- test =
--   toCols $
--   (fromRowMajorVectorToSD2 (V.fromList [0..9]) :: S ('D2 2 5))
--   (fromRowMajorVectorToSD1 (V.fromList [0..9]) :: S ('D1 10))

-- testShapeV = fromRowMajorVectorToSD2V (V.fromList [0..9]) :: S ('D2 2 5)
-- testShape = fromRowMajorVectorToSD2 (V.fromList [0..9]) :: S ('D2 2 5)

-- | Convert the Shape to a list of rows.
toRows :: S x -> [V.Vector RealNum]
toRows x =
  case x of
    (S1DV v)  -> [v]
    (S1D v)   -> [unsafeCoerce v]
    (S2D m)   -> map LAS.extract . LAS.toRows $ m
    (S2DV {}) -> toRowsS2D x
    (S3D m)   -> map LAS.extract . LAS.toRows $ m
{-# INLINE toRows #-}

-- | Convert the Shape to a list of columns.
toCols :: S x -> [V.Vector RealNum]
toCols x =
  case x of
    S1DV v    -> map V.singleton . V.toList $ v
    S1D v     -> map V.singleton . V.toList $ (unsafeCoerce v :: V.Vector RealNum)
    (S2DV {}) -> toColumnsS2D x
    (S2D m)   -> map LAS.extract . LAS.toColumns $ m
    (S3D m)   -> map LAS.extract . LAS.toColumns $ m


-- | Converts the given vector to the correct layer shape.
toLayerShape :: S i -> S x -> S x
toLayerShape x y = case (x, y) of
  (S1D{}, S1DV{}) -> toS1D y
  (S1DV{}, S1D{}) -> fromS1D y
  (S2D{}, S2DV{}) -> toS2D y
  (S2DV{}, S2D{}) -> fromS2D y
  _               -> y
{-# INLINE toLayerShape #-}

-- 1D conversion

-- | Convert to S1D.
toS1D :: S ('D1 l) -> S ('D1 l)
toS1D (S1DV vec) = S1D $ (fromMaybe err . LAS.create $ vec)
  where
    err = error $ "wrong length of vector with " ++ show (V.length vec) ++ " in toS1D "
toS1D x@S1D{} = x
{-# INLINE toS1D #-}

-- | Convert from S1DV. This is the same as @toS1D@.
fromS1DV :: S ('D1 l) -> S ('D1 l)
fromS1DV = toS1D
{-# INLINE fromS1DV #-}

-- | Convert from S1D to S1DV.
fromS1D :: S ('D1 l) -> S ('D1 l)
fromS1D (S1D vec) = S1DV (LAS.extract vec)
fromS1D x@S1DV{}  = x
{-# INLINE fromS1D #-}

-- | Convert to S1DV (from S1D). Thsi is the same as @fromS1D@.
toS1DV :: S ('D1 l) -> S ('D1 l)
toS1DV = fromS1D
{-# INLINE toS1DV #-}


-- 2D conversions

-- | Convert from vector representation.
toS2D :: forall i j . S ('D2 i j) -> S ('D2 i j)
toS2D mat@S2DV{} = S2D $ LAS.matrix (concatMap V.toList $ toRows mat)
toS2D x@S2D{}    = x
{-# INLINE toS2D #-}

-- | Convert from S2DV. This is the same as @toS2D@.
fromS2DV :: S ('D2 i j) -> S ('D2 i j)
fromS2DV = toS2D
{-# INLINE fromS2DV #-}

-- | Convert from S2D to S2DV.
fromS2D :: S ('D2 i j) -> S ('D2 i j)
fromS2D (S2D mat) = S2DV . V.concat . map LAS.extract . LAS.toColumns $ mat
fromS2D x@S2DV{}  = x
{-# INLINE fromS2D #-}

-- | Convert to S2DV. This is the same as @fromS2D@.
toS2DV :: S ('D2 i j) -> S ('D2 i j)
toS2DV = fromS2D
{-# INLINE toS2DV #-}


-- test = LAS.create $ NLA.reshape 5 $ (V.fromList [0..9]) :: Maybe (LAS.L 2 5)
-- Just (matrix
--  [ 0.0, 1.0, 2.0, 3.0, 4.0
--  , 5.0, 6.0, 7.0, 8.0, 9.0 ] :: L 2 5)

-- | Efficiently extract the columns of the shape. Returns a list of j vectors.
toColumnsS2D :: forall i j . S ('D2 i j) -> [V.Vector RealNum]
toColumnsS2D (S2D mat) = map LAS.extract . LAS.toColumns $ mat
toColumnsS2D (S2DV vec) = map (\idx -> V.slice idx m vec) [0,m .. (V.length vec - m)]
  where
    m = fromIntegral $ natVal (Proxy :: Proxy i)
{-# INLINE toColumnsS2D #-}

-- | Efficiently extract the rows of the shape. Returns a list of i vectors.
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

-- | Convert from a row major vector to @SD1@.
fromRowMajorVectorToSD1 :: forall l . (KnownNat l) => V.Vector RealNum -> S ('D1 l)
fromRowMajorVectorToSD1 vec
  | V.length vec /= l = error $ "cannot create Vector R " ++ show l ++ " from vector with length " ++ show (V.length vec) ++ " in fromRowMajorVectorToSD1"
  | otherwise = S1D (unsafeCoerce vec)
  where l = fromIntegral $ natVal (Proxy :: Proxy l)
{-# INLINE fromRowMajorVectorToSD1 #-}

-- | Convert from a row major vector to @SD1V@.
fromRowMajorVectorToSD1V :: forall l . (KnownNat l) => V.Vector RealNum -> S ('D1 l)
fromRowMajorVectorToSD1V = S1DV
{-# INLINE fromRowMajorVectorToSD1V #-}

-- | Convert from a row major vector to @SD2@.
fromRowMajorVectorToSD2 :: forall i j . (KnownNat i, KnownNat j) => V.Vector RealNum -> S ('D2 i j)
fromRowMajorVectorToSD2 vec
  | V.length vec /= m * n = error $ "cannot create matrix L " ++ show (m,n) ++ " from vector length " ++ show (V.length vec) ++ " in fromRowMajorVectorToSD2"
  | otherwise = toS2D $ fromRowMajorVectorToSD2V vec
                -- in S2D $ unsafeCoerce v'
  where
    m = fromIntegral $ natVal (Proxy :: Proxy i)
    n = fromIntegral $ natVal (Proxy :: Proxy j)
{-# INLINE fromRowMajorVectorToSD2 #-}

-- toRowMajorVector :: (KnownNat i, KnownNat j) => S ('D2 i j) -> V.Vector RealNum
-- toRowMajorVector = V.concat . toColumnsS2D

-- | Convert from a row major vector to @SD2V@.
fromRowMajorVectorToSD2V :: forall i j . (KnownNat i, KnownNat j) => V.Vector RealNum -> S ('D2 i j)
fromRowMajorVectorToSD2V vec = unsafePerformIO $ do
  vec' <- createVector (V.length vec)
  V.unsafeWith vec $ \from ->
    V.unsafeWith vec' $ \to ->  do
    let go (-1) = return ()
        go !k = do
          -- let idx = nRow * m + nCol
              -- nCol = k `div` n
              -- nRow = k `mod` n
          let idx = columnMajorModeIndex n row col
              col = k - row * n
              row = k `mod` n
          x <- peekElemOff from k
          pokeElemOff to idx x
          go (k-1)
    go (V.length vec - 1)
  return $ S2DV vec'
  where
    m = fromIntegral $ natVal (Proxy :: Proxy i)
    n = fromIntegral $ natVal (Proxy :: Proxy j)
{-# INLINE fromRowMajorVectorToSD2V #-}


-- | Convert from a row major vector to @SD2V@.
fromColumnMajorVectorToSD2V :: forall i j . (KnownNat i, KnownNat j) => V.Vector RealNum -> S ('D2 i j)
fromColumnMajorVectorToSD2V = S2DV
{-# INLINE fromColumnMajorVectorToSD2V #-}


test =
  -- (\x -> fromColumnMajorVectorToSD2V x :: S ('D2 5 2)) $
  -- V.concat $
  -- toRowsS2D $
  -- ttoRowMajorVector $
  fromS2D $
  -- (\x -> fromRowMajorVectorToSD2V x :: S ('D2 5 2) ) $ V.concat $
  -- toRowsS2D $
  -- ((\(S2D x) -> S2DV $ V.concat $ map LAS.extract . LAS.toColumns $ x :: S ('D2 2 5)) )
  -- (fromRowMajorVectorToSD2V (V.fromList [0..9]) :: S ('D2 2 5))
  (fromRowMajorVectorToSD2 (V.fromList [0..9]) :: S ('D2 5 2))
  -- (fromRowMajorVectorToSD1 (V.fromList [0..9]) :: S ('D1 10))
