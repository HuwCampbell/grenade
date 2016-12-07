module Grenade.Layers.Convolution.Internal (
    im2col
--   , im2colUnsafe
  , vid2col
  , col2im
  , col2imFit
  , col2vid

  , col2vidUnsafe
  , col2imUnsafe
  , im2colUnsafe
  , vid2colUnsafe
  , fittingStarts
  ) where

import           Control.Monad.ST
import           Control.Parallel.Strategies ( parMap, rseq )

import           Data.STRef
import           Data.Foldable ( forM_ )

import           Numeric.LinearAlgebra hiding ( uniformSample, konst )
import qualified Numeric.LinearAlgebra as LA
import qualified Numeric.LinearAlgebra.Devel as U

im2col :: Int -> Int -> Int -> Int -> Matrix Double -> Matrix Double
im2col nrows ncols srows scols m =
  let starts = fittingStarts (rows m) nrows srows (cols m) ncols scols
  in  im2colFit starts nrows ncols m

im2colFit :: [(Int,Int)] -> Int -> Int -> Matrix Double -> Matrix Double
im2colFit starts nrows ncols m =
  let imRows = fmap (\start -> flatten $ subMatrix start (nrows, ncols) m) starts
  in  fromRows imRows

vid2col :: Int -> Int -> Int -> Int -> Int -> Int -> [Matrix Double] -> Matrix Double
vid2col nrows ncols srows scols inputrows inputcols ms =
  let starts = fittingStarts inputrows nrows srows inputcols ncols scols
      subs   = parMap rseq (im2colFit starts nrows ncols) ms
  in  foldl1 (|||) subs

col2vid :: Int -> Int -> Int -> Int -> Int -> Int -> Matrix Double -> [Matrix Double]
col2vid krows kcols srows scols drows dcols m =
  let starts = fittingStart (cols m) (krows * kcols) (krows * kcols)
      r      = rows m
      mats   = fmap (\s -> subMatrix (0,s) (r, krows * kcols) m) starts
      colSts = fittingStarts drows krows srows dcols kcols scols
  in  parMap rseq (col2imFit colSts krows kcols drows dcols) mats

col2im :: Int -> Int -> Int -> Int -> Int -> Int -> Matrix Double -> Matrix Double
col2im krows kcols srows scols drows dcols m =
  let starts     = fittingStarts drows krows srows dcols kcols scols
  in  col2imFit starts krows kcols drows dcols m

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
              = error "Kernel and step do not fit in matrix."
  in  go 0

col2imFit :: [(Int,Int)] -> Int -> Int -> Int -> Int -> Matrix Double -> Matrix Double
col2imFit starts krows kcols drows dcols m =
  let indicies = (\[a,b] -> (a,b)) <$> sequence [[0..(krows-1)], [0..(kcols-1)]]
      convs      = fmap (zip indicies . toList) . toRows $ m
      pairs      = zip convs starts
      accums     = concatMap (\(conv',(stx',sty')) -> fmap (\((ix,iy), val) -> ((ix + stx', iy + sty'), val)) conv') pairs
  in  accum (LA.konst 0 (drows, dcols)) (+) accums

-- void col2im_cpu(const Dtype* data_col, const int channels,
--     const int height, const int width, const int kernel_h, const int kernel_w,
--     const int pad_h, const int pad_w,
--     const int stride_h, const int stride_w,
--     const int dilation_h, const int dilation_w,
--     Dtype* data_im) {
--   caffe_set(height * width * channels, Dtype(0), data_im);
--   const int output_h = (height + 2 * pad_h -
--     (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
--   const int output_w = (width + 2 * pad_w -
--     (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
--   const int channel_size = height * width;
--   for (int channel = channels; channel--; data_im += channel_size) {
--     for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
--       for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
--         int input_row = -pad_h + kernel_row * dilation_h;
--         for (int output_rows = output_h; output_rows; output_rows--) {
--           if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
--             data_col += output_w;
--           } else {
--             int input_col = -pad_w + kernel_col * dilation_w;
--             for (int output_col = output_w; output_col; output_col--) {
--               if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
--                 data_im[input_row * width + input_col] += *data_col;
--               }
--               data_col++;
--               input_col += stride_w;
--             }
--           }
--           input_row += stride_h;
--         }
--       }
--     }
--   }
-- }


--   let starts = fittingStart (cols m) (krows * kcols) (krows * kcols)
--       r      = rows m
--       mats   = fmap (\s -> subMatrix (0,s) (r, krows * kcols) m) starts
--   in  parMap rseq (col2imUnsafe krows kcols srows scols drows dcols) mats

col2imUnsafe :: Int -> Int -> Int -> Int -> Int -> Int -> Matrix Double -> Matrix Double
col2imUnsafe kernelRows kernelColumns strideRows strideColumns destinationRows destinationCols columnMatrix = U.runSTMatrix $ do
  let columnMatrixRows = rows columnMatrix

  dataIm  <- U.newMatrix 0 destinationRows destinationCols

  offsetR <- newSTRef 0
  offsetC <- newSTRef 0

  forM_ [0 .. columnMatrixRows - 1] $ \ir -> do
    inputColumn <- newSTRef 0
    forM_ [0 .. kernelRows -1] $ \kr ->
      forM_ [0 .. kernelColumns -1] $ \kc -> do
        ic       <- readSTRef inputColumn
        offsetR' <- readSTRef offsetR
        offsetC' <- readSTRef offsetC
        U.modifyMatrix dataIm (kr + offsetR') (kc + offsetC') (+ atIndex columnMatrix (ir,ic))
        modifySTRef inputColumn (+1)

    offsetC' <- readSTRef offsetC
    if offsetC' + kernelColumns < destinationCols
      then modifySTRef offsetC (+ strideColumns)
      else writeSTRef offsetC 0 >> modifySTRef offsetR (+ strideRows)

  return dataIm

col2vidUnsafe :: Int -> Int -> Int -> Int -> Int -> Int -> Matrix Double -> [Matrix Double]
col2vidUnsafe kernelRows kernelColumns strideRows strideColumns destinationRows destinationCols columnMatrix = runST $ do
  let columnMatrixRows = rows columnMatrix
  let filters = cols columnMatrix `div` (kernelRows * kernelColumns)

  dataIms    <- traverse (\_ -> U.newMatrix 0 destinationRows destinationCols) [0 .. filters-1]

  offsetR    <- newSTRef 0
  offsetC    <- newSTRef 0
  offsetM    <- newSTRef 0

  forM_ dataIms $ \dataIm -> do
    offsetM' <- readSTRef offsetM
    forM_ [0 .. columnMatrixRows - 1] $ \ir -> do
      inputColumn <- newSTRef 0
      forM_ [0 .. kernelRows -1] $ \kr ->
        forM_ [0 .. kernelColumns -1] $ \kc -> do
          ic       <- readSTRef inputColumn
          offsetR' <- readSTRef offsetR
          offsetC' <- readSTRef offsetC
          U.modifyMatrix dataIm (kr + offsetR') (kc + offsetC') (+ atIndex columnMatrix (ir, ic + offsetM'))
          modifySTRef inputColumn (+1)

      offsetC' <- readSTRef offsetC
      if offsetC' + kernelColumns < destinationCols
        then modifySTRef offsetC (+ strideColumns)
        else writeSTRef offsetC 0 >> modifySTRef offsetR (+ strideRows)

    writeSTRef offsetR 0
    writeSTRef offsetC 0
    modifySTRef offsetM (+ (kernelRows * kernelColumns))

  traverse U.freezeMatrix dataIms

vid2colUnsafe :: Int -> Int -> Int -> Int -> Int -> Int -> Int -> [Matrix Double] -> Matrix Double
vid2colUnsafe channels kernelRows kernelColumns striderows stridecols vidrows vidcols dataVid = U.runSTMatrix $ do
  let starts          = fittingStarts vidrows kernelRows striderows vidcols kernelColumns stridecols
      matWidth        = kernelRows * kernelColumns
      destinationRows = 1 + (vidrows - kernelRows) `div` striderows
      destinationCols = 1 + (vidcols - kernelColumns) `div` stridecols
      destinationSize = destinationRows * destinationCols

  dataCol <- U.newMatrix 0 destinationSize (channels * matWidth)

  offsetC <- newSTRef 0

  forM_ dataVid $ \dataIm -> do
    inputRow <- newSTRef 0
    offsetC'     <- readSTRef offsetC
    forM_ starts $ \(startRow, startCol) -> do
      inputColumn <- newSTRef 0
      inputRow'   <- readSTRef inputRow
      forM_ [0 .. kernelRows -1] $ \kr ->
        forM_ [0 .. kernelColumns -1] $ \kc -> do
          inputColumn' <- readSTRef inputColumn
          U.modifyMatrix dataCol inputRow' (inputColumn' + offsetC') (+ atIndex dataIm (kr + startRow, kc + startCol))
          modifySTRef inputColumn (+1)
      modifySTRef inputRow (+1)

    modifySTRef offsetC (+ matWidth)

  return dataCol

im2colUnsafe :: Int -> Int -> Int -> Int -> Int -> Int -> Matrix Double -> Matrix Double
im2colUnsafe kernelRows kernelColumns striderows stridecols vidrows vidcols dataIm = U.runSTMatrix $ do
  let starts          = fittingStarts vidrows kernelRows striderows vidcols kernelColumns stridecols
      matWidth        = kernelRows * kernelColumns
      destinationRows = 1 + (vidrows - kernelRows) `div` striderows
      destinationCols = 1 + (vidcols - kernelColumns) `div` stridecols
      destinationSize = destinationRows * destinationCols

  dataCol <- U.newMatrix 0 destinationSize matWidth

  inputRow <- newSTRef 0
  forM_ starts $ \(startRow, startCol) -> do
    inputColumn <- newSTRef 0
    inputRow'   <- readSTRef inputRow
    forM_ [0 .. kernelRows -1] $ \kr ->
      forM_ [0 .. kernelColumns -1] $ \kc -> do
        inputColumn' <- readSTRef inputColumn
        U.modifyMatrix dataCol inputRow' inputColumn' (+ atIndex dataIm (kr + startRow, kc + startCol))
        modifySTRef inputColumn (+1)
    modifySTRef inputRow (+1)

  return dataCol
