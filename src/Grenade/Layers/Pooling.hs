{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE StandaloneDeriving    #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE PolyKinds             #-}

module Grenade.Layers.Pooling (
    Pooling (..)
  , poolForward
  , poolBackward
  ) where

import           Data.Maybe
import           Data.Proxy
import           Data.Singletons.TypeLits
import           GHC.TypeLits

import           Grenade.Core.Network
import           Grenade.Core.Shape
import           Grenade.Core.Vector
import           Grenade.Layers.Convolution

import           Numeric.LinearAlgebra hiding (uniformSample)
import qualified Numeric.LinearAlgebra as LA
import           Numeric.LinearAlgebra.Static as LAS hiding ((|||), build, toRows)

-- | A pooling layer for a neural network.
--   Does a max pooling, looking over a kernel similarly to the convolution network, but returning
--   maxarg only. This layer is often used to provide minor amounts of translational invariance.
--
--   The kernel size dictates which input and output sizes will "fit". Fitting the equation:
--   `out = (in - kernel) / stride + 1` for both dimensions.
--
data Pooling :: Nat
             -> Nat
             -> Nat
             -> Nat -> * where
  Pooling :: ( KnownNat kernelRows
             , KnownNat kernelColumns
             , KnownNat strideRows
             , KnownNat strideColumns
             ) => Pooling kernelRows kernelColumns strideRows strideColumns

instance Show (Pooling k k' s s') where
  show Pooling = "Pooling"


-- | A two dimentional image can be pooled.
instance ( Monad m
         , KnownNat kernelRows
         , KnownNat kernelColumns
         , KnownNat strideRows
         , KnownNat strideColumns
         , KnownNat inputRows
         , KnownNat inputColumns
         , KnownNat outputRows
         , KnownNat outputColumns
         , ((outputRows - 1) * strideRows) ~ (inputRows - kernelRows)
         , ((outputColumns - 1) * strideColumns) ~ (inputColumns - kernelColumns)
         ) => Layer m (Pooling kernelRows kernelColumns strideRows strideColumns) ('D2 inputRows inputColumns) ('D2 outputRows outputColumns) where
  runForwards Pooling (S2D' input) =
    let kx = fromIntegral $ natVal (Proxy :: Proxy kernelRows)
        ky = fromIntegral $ natVal (Proxy :: Proxy kernelColumns)
        sx = fromIntegral $ natVal (Proxy :: Proxy strideRows)
        sy = fromIntegral $ natVal (Proxy :: Proxy strideColumns)
        ox = fromIntegral $ natVal (Proxy :: Proxy outputRows)
        oy = fromIntegral $ natVal (Proxy :: Proxy outputColumns)
        ex = extract input
        r  = poolForward kx ky sx sy ox oy $ ex
        rs = fromJust . create $ r
    in  return . S2D' $ rs
  runBackards _ Pooling (S2D' input) (S2D' dEdy) =
    let kx = fromIntegral $ natVal (Proxy :: Proxy kernelRows)
        ky = fromIntegral $ natVal (Proxy :: Proxy kernelColumns)
        sx = fromIntegral $ natVal (Proxy :: Proxy strideRows)
        sy = fromIntegral $ natVal (Proxy :: Proxy strideColumns)
        ex = extract input
        eo = extract dEdy
        vs = poolBackward kx ky sx sy ex eo
    in  return (Pooling, S2D' . fromJust . create $ vs)


-- | A three dimensional image can be pooled on each layer.
instance ( Monad m
         , KnownNat kernelRows
         , KnownNat kernelColumns
         , KnownNat strideRows
         , KnownNat strideColumns
         , KnownNat inputRows
         , KnownNat inputColumns
         , KnownNat outputRows
         , KnownNat outputColumns
         , ((outputRows - 1) * strideRows) ~ (inputRows - kernelRows)
         , ((outputColumns - 1) * strideColumns) ~ (inputColumns - kernelColumns)
         ) => Layer m (Pooling kernelRows kernelColumns strideRows strideColumns) ('D3 inputRows inputColumns channels) ('D3 outputRows outputColumns channels) where
  runForwards Pooling (S3D' input) =
    let ix = fromIntegral $ natVal (Proxy :: Proxy inputRows)
        iy = fromIntegral $ natVal (Proxy :: Proxy inputColumns)
        kx = fromIntegral $ natVal (Proxy :: Proxy kernelRows)
        ky = fromIntegral $ natVal (Proxy :: Proxy kernelColumns)
        sx = fromIntegral $ natVal (Proxy :: Proxy strideRows)
        sy = fromIntegral $ natVal (Proxy :: Proxy strideColumns)
        ox = fromIntegral $ natVal (Proxy :: Proxy outputRows)
        oy = fromIntegral $ natVal (Proxy :: Proxy outputColumns)
        ex = fmap extract input
        r  = poolForwardList kx ky sx sy ix iy ox oy ex
        rs = fmap (fromJust . create) r
    in  return . S3D' $ rs
  runBackards _ Pooling (S3D' input) (S3D' dEdy) =
    let ix = fromIntegral $ natVal (Proxy :: Proxy inputRows)
        iy = fromIntegral $ natVal (Proxy :: Proxy inputColumns)
        kx = fromIntegral $ natVal (Proxy :: Proxy kernelRows)
        ky = fromIntegral $ natVal (Proxy :: Proxy kernelColumns)
        sx = fromIntegral $ natVal (Proxy :: Proxy strideRows)
        sy = fromIntegral $ natVal (Proxy :: Proxy strideColumns)
        ex = fmap extract input
        eo = fmap extract dEdy
        ez = vectorZip (,) ex eo
        vs = poolBackwardList kx ky sx sy ix iy ez
    in  return (Pooling, S3D' . fmap (fromJust . create) $ vs)

poolForward :: Int -> Int -> Int -> Int -> Int -> Int -> Matrix Double -> Matrix Double
poolForward nrows ncols srows scols outputRows outputCols m =
  let starts = fittingStarts (rows m) nrows srows (cols m) ncols scols
  in  poolForwardFit starts nrows ncols outputRows outputCols m

poolForwardList :: Functor f => Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> f (Matrix Double) -> f (Matrix Double)
poolForwardList nrows ncols srows scols inRows inCols outputRows outputCols ms =
  let starts = fittingStarts inRows nrows srows inCols ncols scols
  in  poolForwardFit starts nrows ncols outputRows outputCols <$> ms

poolForwardFit :: [(Int,Int)] -> Int -> Int -> Int -> Int -> Matrix Double -> Matrix Double
poolForwardFit starts nrows ncols _ outputCols m =
  let els    = fmap (\start -> maxElement $ subMatrix start (nrows, ncols) m) starts
  in  LA.matrix outputCols els

poolBackward :: Int -> Int -> Int -> Int -> Matrix Double -> Matrix Double -> Matrix Double
poolBackward krows kcols srows scols inputMatrix gradientMatrix =
  let inRows     = (rows inputMatrix)
      inCols     = (cols inputMatrix)
      starts     = fittingStarts inRows krows srows inCols kcols scols
  in  poolBackwardFit starts krows kcols inputMatrix gradientMatrix

poolBackwardList :: Functor f => Int -> Int -> Int -> Int -> Int -> Int -> f (Matrix Double, Matrix Double) -> f (Matrix Double)
poolBackwardList krows kcols srows scols inRows inCols inputMatrices =
  let starts     = fittingStarts inRows krows srows inCols kcols scols
  in  (uncurry $ poolBackwardFit starts krows kcols) <$> inputMatrices

poolBackwardFit :: [(Int,Int)] -> Int -> Int -> Matrix Double -> Matrix Double -> Matrix Double
poolBackwardFit starts krows kcols inputMatrix gradientMatrix =
  let inRows     = (rows inputMatrix)
      inCols     = (cols inputMatrix)
      inds       = fmap (\start -> maxIndex $ subMatrix start (krows, kcols) inputMatrix) starts
      grads      = toList $ flatten gradientMatrix
      grads'     = zip3 starts grads inds
      accums     = fmap (\((stx',sty'),grad,(inx, iny)) -> ((stx' + inx, sty' + iny), grad)) grads'
  in  accum (LA.konst 0 (inRows, inCols)) (+) accums
