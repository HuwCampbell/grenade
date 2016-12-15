module Test.Grenade.Layers.Internal.Reference where

import           Numeric.LinearAlgebra

im2col :: Int -> Int -> Int -> Int -> Matrix Double -> Matrix Double
im2col nrows ncols srows scols m =
  let starts = fittingStarts (rows m) nrows srows (cols m) ncols scols
  in  im2colFit starts nrows ncols m

vid2col :: Int -> Int -> Int -> Int -> Int -> Int -> [Matrix Double] -> Matrix Double
vid2col nrows ncols srows scols inputrows inputcols ms =
  let starts = fittingStarts inputrows nrows srows inputcols ncols scols
      subs   = fmap (im2colFit starts nrows ncols) ms
  in  foldl1 (|||) subs

im2colFit :: [(Int,Int)] -> Int -> Int -> Matrix Double -> Matrix Double
im2colFit starts nrows ncols m =
  let imRows = fmap (\start -> flatten $ subMatrix start (nrows, ncols) m) starts
  in  fromRows imRows

col2vid :: Int -> Int -> Int -> Int -> Int -> Int -> Matrix Double -> [Matrix Double]
col2vid nrows ncols srows scols drows dcols m =
  let starts = fittingStart (cols m) (nrows * ncols) (nrows * ncols)
      r      = rows m
      mats   = fmap (\s -> subMatrix (0,s) (r, nrows * ncols) m) starts
      colSts = fittingStarts drows nrows srows dcols ncols scols
  in  fmap (col2imfit colSts nrows ncols drows dcols) mats

col2im :: Int -> Int -> Int -> Int -> Int -> Int -> Matrix Double -> Matrix Double
col2im krows kcols srows scols drows dcols m =
  let starts     = fittingStarts drows krows srows dcols kcols scols
  in  col2imfit starts krows kcols drows dcols m

col2imfit :: [(Int,Int)] -> Int -> Int -> Int -> Int -> Matrix Double -> Matrix Double
col2imfit starts krows kcols drows dcols m =
  let indicies   = (\[a,b] -> (a,b)) <$> sequence [[0..(krows-1)], [0..(kcols-1)]]
      convs      = fmap (zip indicies . toList) . toRows $ m
      pairs      = zip convs starts
      accums     = concatMap (\(conv',(stx',sty')) -> fmap (\((ix,iy), val) -> ((ix + stx', iy + sty'), val)) conv') pairs
  in  accum (konst 0 (drows, dcols)) (+) accums

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
  in  matrix outputCols els

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
  in  accum (konst 0 (inRows, inCols)) (+) accums

-- | These functions are not even remotely safe, but it's only called from the statically typed
--   commands, so we should be good ?!?!?
--   Returns the starting sub matrix locations which fit inside the larger matrix for the
--   convolution. Takes into account the stride and kernel size.
fittingStarts :: Int -> Int -> Int -> Int -> Int -> Int -> [(Int,Int)]
fittingStarts nrows kernelrows steprows ncols kernelcols stepcolsh =
  let rs = fittingStart nrows kernelrows steprows
      cs = fittingStart ncols kernelcols stepcolsh
      ls = sequence [rs, cs]
  in  fmap (\[a,b] -> (a,b)) ls

-- | Returns the starting sub vector which fit inside the larger vector for the
--   convolution. Takes into account the stride and kernel size.
fittingStart :: Int -> Int -> Int -> [Int]
fittingStart width kernel steps =
  let go left | left + kernel < width
              = left : go (left + steps)
              | left + kernel == width
              = [left]
              | otherwise
              = []
  in go 0
