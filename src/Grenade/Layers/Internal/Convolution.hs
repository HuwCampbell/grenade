{-# LANGUAGE ForeignFunctionInterface #-}
module Grenade.Layers.Internal.Convolution (
    col2vidUnsafe
  , col2imUnsafe
  , vid2colUnsafe
  , im2colUnsafe
  , im2col
  , col2im
  , col2vid
  , vid2col
  , fittingStarts
  , unsafeModifyMatrix
  ) where

import           Control.Monad.ST ( ST, runST )

import           Data.STRef ( newSTRef, modifySTRef', writeSTRef, readSTRef )
import           Data.Foldable ( forM_ )
import           Data.Traversable ( forM )

import           Foreign ( mallocForeignPtrArray0, withForeignPtr )
import           Foreign.Ptr ( Ptr )
import           Foreign.Storable( Storable )

import           Numeric.LinearAlgebra hiding ( uniformSample, konst )
import qualified Numeric.LinearAlgebra.Devel as U

import           System.IO.Unsafe ( unsafePerformIO )

-- This module provides provides im2col function and friends, ala caffe.
--
-- /* From Caffe */
-- @
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
-- @
--

col2vid :: Int -> Int -> Int -> Int -> Int -> Int -> Matrix Double -> [Matrix Double]
col2vid kernelRows kernelColumns strideRows strideColumns height width dataCol =
  let channels = cols dataCol `div` (kernelRows * kernelColumns)
      retMat  = col2im_c channels height width kernelRows kernelColumns strideRows strideColumns dataCol
  in  (\f -> subMatrix (f * height, 0) (height, width) retMat) <$> [0..channels -1]

col2im :: Int -> Int -> Int -> Int -> Int -> Int -> Matrix Double -> Matrix Double
col2im kernelRows kernelColumns strideRows strideColumns height width dataCol =
  let channels = 1
  in  col2im_c channels height width kernelRows kernelColumns strideRows strideColumns dataCol

col2im_c :: Int -> Int -> Int -> Int -> Int -> Int -> Int -> Matrix Double -> Matrix Double
col2im_c channels height width kernelRows kernelColumns strideRows strideColumns dataCol =
  let vec = flatten dataCol
  in unsafePerformIO $ do
    outPtr <- mallocForeignPtrArray0 (height * width * channels)
    let (inPtr, inOffset, _) = U.unsafeToForeignPtr vec

    withForeignPtr inPtr $ \inPtr' ->
      withForeignPtr outPtr $ \outPtr' ->
        col2im_cpu inPtr' inOffset channels height width kernelRows kernelColumns strideRows strideColumns outPtr'

    let matVec = U.unsafeFromForeignPtr outPtr 0 (height * width * channels)
    return $ U.matrixFromVector U.RowMajor (height * channels) width matVec

foreign import ccall safe
    col2im_cpu
      :: Ptr Double -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Ptr Double -> IO ()


-- | col2im function.
--
-- Takes a column patch, and reconstitutes it into a normal image.
-- Does not do any bounds checking on the matrix, so should only
-- be called once the sizes are ensured correct.
col2imUnsafe :: Int -> Int -> Int -> Int -> Int -> Int -> Matrix Double -> Matrix Double
col2imUnsafe kernelRows kernelColumns strideRows strideColumns destinationRows destinationCols columnMatrix = U.runSTMatrix $ do
  let columnMatrixRows = rows columnMatrix

  dataIm  <- U.newMatrix 0 destinationRows destinationCols
  offsetR <- newSTRef 0
  offsetC <- newSTRef 0

  forM_ [0 .. columnMatrixRows - 1] $ \inputRow -> do
    inputColumnRef <- newSTRef 0
    offsetR' <- readSTRef offsetR
    offsetC' <- readSTRef offsetC
    forM_ [offsetR' .. offsetR' + kernelRows -1] $ \kr ->
      forM_ [offsetC' .. offsetC' + kernelColumns -1] $ \kc -> do
        inputColumn <- readSTRef inputColumnRef
        unsafeModifyMatrix dataIm kr kc (+ U.atM' columnMatrix inputRow inputColumn)
        modifySTRef' inputColumnRef (+1)

    if offsetC' + kernelColumns < destinationCols
      then modifySTRef' offsetC (+ strideColumns)
      else writeSTRef offsetC 0 >> modifySTRef' offsetR (+ strideRows)

  return dataIm

-- | col2vid function.
--
-- Takes a column patch image, and reconstitutes it into a normal image with multiple channels.
-- Does not do any bounds checking on the matrix, so should only
-- be called once the sizes are ensured correct.
col2vidUnsafe :: Int -> Int -> Int -> Int -> Int -> Int -> Matrix Double -> [Matrix Double]
col2vidUnsafe kernelRows kernelColumns strideRows strideColumns destinationRows destinationCols columnMatrix = runST $ do
  let columnMatrixRows = rows columnMatrix
  let filters = cols columnMatrix `div` (kernelRows * kernelColumns)

  forM [0 .. filters - 1] $ \iter -> do
    let offsetM = iter * (kernelRows * kernelColumns)
    dataIm     <- U.newMatrix 0 destinationRows destinationCols
    offsetR    <- newSTRef 0
    offsetC    <- newSTRef 0
    forM_ [0 .. columnMatrixRows - 1] $ \ir -> do
      inputColumn <- newSTRef offsetM
      offsetR'    <- readSTRef offsetR
      offsetC'    <- readSTRef offsetC
      forM_ [offsetR' .. offsetR' + kernelRows -1] $ \kr ->
        forM_ [offsetC' .. offsetC' + kernelColumns -1] $ \kc -> do
          ic       <- readSTRef inputColumn
          unsafeModifyMatrix dataIm kr kc (+ U.atM' columnMatrix ir ic)
          modifySTRef' inputColumn (+1)

      if offsetC' + kernelColumns < destinationCols
        then modifySTRef' offsetC (+ strideColumns)
        else writeSTRef offsetC 0 >> modifySTRef' offsetR (+ strideRows)

    U.unsafeFreezeMatrix dataIm

vid2col :: Int -> Int -> Int -> Int -> Int -> Int -> [Matrix Double] -> Matrix Double
vid2col  kernelRows kernelColumns strideRows strideColumns height width dataVid =
  let channels = length dataVid
  in  im2col_c channels height width kernelRows kernelColumns strideRows strideColumns (foldl1 (===) dataVid)


im2col :: Int -> Int -> Int -> Int -> Matrix Double -> Matrix Double
im2col kernelRows kernelColumns strideRows strideColumns dataIm =
  let channels = 1
      height = rows dataIm
      width  = cols dataIm
  in  im2col_c channels height width kernelRows kernelColumns strideRows strideColumns dataIm

im2col_c :: Int -> Int -> Int -> Int -> Int -> Int -> Int -> Matrix Double -> Matrix Double
im2col_c channels height width kernelRows kernelColumns strideRows strideColumns dataIm =
  let vec    = flatten dataIm
      rowOut = (height - kernelRows) `div` strideRows + 1
      colOut = (width - kernelColumns) `div` strideColumns + 1
      kernelSize      = kernelRows * kernelColumns
      numberOfPatches = rowOut * colOut
  in unsafePerformIO $ do
    outPtr <- mallocForeignPtrArray0 (numberOfPatches * kernelSize * channels)
    let (inPtr, inOffset, _) = U.unsafeToForeignPtr vec

    withForeignPtr inPtr $ \inPtr' ->
      withForeignPtr outPtr $ \outPtr' ->
        im2col_cpu inPtr' inOffset channels height width kernelRows kernelColumns strideRows strideColumns outPtr'

    let matVec = U.unsafeFromForeignPtr outPtr 0 (numberOfPatches * kernelSize * channels)
    return $ U.matrixFromVector U.RowMajor numberOfPatches (kernelSize * channels) matVec

foreign import ccall safe
    im2col_cpu
      :: Ptr Double -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Ptr Double -> IO ()


unsafeModifyMatrix :: (Storable t) => U.STMatrix s t -> Int -> Int -> (t -> t) -> ST s ()
unsafeModifyMatrix x r c f = U.unsafeReadMatrix x r c >>= U.unsafeWriteMatrix x r c . f
{-# INLINE unsafeModifyMatrix #-}


-- | Returns the starting sub matrix locations which fit inside the larger matrix for the
--   convolution. Takes into account the stride and kernel size.
fittingStarts :: Int -> Int -> Int -> Int -> Int -> Int -> [(Int,Int)]
fittingStarts nrows kernelrows steprows ncols kernelcols stepcolsh =
  let rs = fittingStart nrows kernelrows steprows
      cs = fittingStart ncols kernelcols stepcolsh
  in  concatMap ( \r -> fmap (\c -> (r , c)) cs ) rs

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
  in  go 0




-- | Old functions (useful for sanity checking and benchmarking)

vid2colUnsafe :: Int -> Int -> Int -> Int -> Int -> Int -> [Matrix Double] -> Matrix Double
vid2colUnsafe kernelRows kernelColumns striderows stridecols vidrows vidcols dataVid = U.runSTMatrix $ do
  let starts          = fittingStarts vidrows kernelRows striderows vidcols kernelColumns stridecols
      kernelSize      = kernelRows * kernelColumns
      numberOfPatches = length starts
      channels        = length dataVid

  dataCol <- U.newUndefinedMatrix U.RowMajor numberOfPatches (channels * kernelSize)

  offsetC <- newSTRef 0

  forM_ dataVid $ \dataIm -> do
    inputRowRef  <- newSTRef 0
    offsetC'     <- readSTRef offsetC
    forM_ starts $ \(startRow, startCol) -> do
      inputColumnRef <- newSTRef 0
      inputRow       <- readSTRef inputRowRef
      forM_ [startRow .. startRow + kernelRows -1] $ \kr ->
        forM_ [startCol .. startCol + kernelColumns -1] $ \kc -> do
          inputColumn <- readSTRef inputColumnRef
          U.unsafeWriteMatrix dataCol inputRow (inputColumn + offsetC') (U.atM' dataIm kr kc)
          modifySTRef' inputColumnRef (+1)
      modifySTRef' inputRowRef (+1)

    modifySTRef' offsetC (+ kernelSize)

  return dataCol

im2colUnsafe :: Int -> Int -> Int -> Int -> Matrix Double -> Matrix Double
im2colUnsafe kernelRows kernelColumns striderows stridecols dataIm = U.runSTMatrix $ do
  let starts          = fittingStarts (rows dataIm) kernelRows striderows (cols dataIm) kernelColumns stridecols
      kernelSize      = kernelRows * kernelColumns
      numberOfPatches = length starts

  dataCol <- U.newUndefinedMatrix U.RowMajor numberOfPatches kernelSize

  inputRowRef <- newSTRef 0
  forM_ starts $ \(startRow, startCol) -> do
    inputColumnRef <- newSTRef 0
    inputRow       <- readSTRef inputRowRef
    forM_ [startRow .. startRow + kernelRows -1] $ \kr ->
      forM_ [startCol .. startCol + kernelColumns -1] $ \kc -> do
        inputColumn <- readSTRef inputColumnRef
        U.unsafeWriteMatrix dataCol inputRow inputColumn (U.atM' dataIm kr kc)
        modifySTRef' inputColumnRef (+1)
    modifySTRef' inputRowRef (+1)

  return dataCol


{-# ANN module "HLint: ignore Reduce duplication" #-}
