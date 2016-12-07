module Grenade.Layers.Internal.Pooling (
    poolForward
  , poolBackward
  , poolForwardList
  , poolBackwardList
  ) where

import           Numeric.LinearAlgebra hiding ( uniformSample, konst )
import qualified Numeric.LinearAlgebra as LA

import           Grenade.Layers.Internal.Convolution

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
  let inRows     = rows inputMatrix
      inCols     = cols inputMatrix
      starts     = fittingStarts inRows krows srows inCols kcols scols
  in  poolBackwardFit starts krows kcols inputMatrix gradientMatrix

poolBackwardList :: Functor f => Int -> Int -> Int -> Int -> Int -> Int -> f (Matrix Double, Matrix Double) -> f (Matrix Double)
poolBackwardList krows kcols srows scols inRows inCols inputMatrices =
  let starts     = fittingStarts inRows krows srows inCols kcols scols
  in  uncurry (poolBackwardFit starts krows kcols) <$> inputMatrices

poolBackwardFit :: [(Int,Int)] -> Int -> Int -> Matrix Double -> Matrix Double -> Matrix Double
poolBackwardFit starts krows kcols inputMatrix gradientMatrix =
  let inRows     = rows inputMatrix
      inCols     = cols inputMatrix
      inds       = fmap (\start -> maxIndex $ subMatrix start (krows, kcols) inputMatrix) starts
      grads      = toList $ flatten gradientMatrix
      grads'     = zip3 starts grads inds
      accums     = fmap (\((stx',sty'),grad,(inx, iny)) -> ((stx' + inx, sty' + iny), grad)) grads'
  in  accum (LA.konst 0 (inRows, inCols)) (+) accums
