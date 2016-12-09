module Grenade.Layers.Internal.Pooling (
    poolForward
  , poolBackward
  , poolForwardList
  , poolBackwardList
  ) where

import           Data.Foldable ( forM_ )
import           Data.Function ( on )
import           Data.List ( maximumBy )

import           Numeric.LinearAlgebra hiding ( uniformSample, konst )
import qualified Numeric.LinearAlgebra as LA
import qualified Numeric.LinearAlgebra.Devel as U

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
  let els    = fmap (\start -> unsafeMaxElement $ subMatrix start (nrows, ncols) m) starts
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

-- poolBackwardFit :: [(Int,Int)] -> Int -> Int -> Matrix Double -> Matrix Double -> Matrix Double
-- poolBackwardFit starts krows kcols inputMatrix gradientMatrix =
--   let inRows     = rows inputMatrix
--       inCols     = cols inputMatrix
--       inds       = fmap (\start -> unsafeMaxIndex $ subMatrix start (krows, kcols) inputMatrix) starts
--       grads      = toList $ flatten gradientMatrix
--       grads'     = zip3 starts grads inds
--       accums     = fmap (\((stx',sty'),grad,(inx, iny)) -> ((stx' + inx, sty' + iny), grad)) grads'
--   in  accum (LA.konst 0 (inRows, inCols)) (+) accums

poolBackwardFit :: [(Int,Int)] -> Int -> Int -> Matrix Double -> Matrix Double -> Matrix Double
poolBackwardFit starts krows kcols inputMatrix gradientMatrix = U.runSTMatrix $ do
  let inRows     = rows inputMatrix
      inCols     = cols inputMatrix
      gradCol    = cols gradientMatrix
      extent     = (krows, kcols)

  retM <- U.newMatrix 0 inRows inCols

  forM_ (zip [0..] starts) $ \(ix, start) -> do
    let loc = unsafeMaxIndexSubMatrix start extent inputMatrix
    uncurry (unsafeModifyMatrix retM) loc ((+) $ uncurry (U.atM' gradientMatrix) $ divMod ix gradCol)

  return retM

unsafeMaxElement :: Matrix Double -> Double
unsafeMaxElement m = uncurry (U.atM' m) $ unsafeMaxIndex m

unsafeMaxIndex :: Matrix Double -> (Int, Int)
unsafeMaxIndex m =
  let mrows = [0 .. rows m - 1]
      mcols = [0 .. cols m - 1]
      pairs = concatMap ( \r -> fmap (\c -> (r , c)) mcols ) mrows
  in  maximumBy (compare `on` uncurry (U.atM' m)) pairs


unsafeMaxIndexSubMatrix :: (Int,Int) -> (Int,Int) -> Matrix Double -> (Int, Int)
unsafeMaxIndexSubMatrix  (startRow, startCol) (extentRow, extentCold) m =
  let mrows = [startRow .. startRow + extentRow  - 1]
      mcols = [startCol .. startCol + extentCold - 1]
      pairs = concatMap ( \r -> fmap (\c -> (r , c)) mcols ) mrows
  in  maximumBy (compare `on` uncurry (U.atM' m)) pairs

