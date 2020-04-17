{-# LANGUAGE ForeignFunctionInterface #-}
module Grenade.Layers.Internal.Convolution (
    im2col
  , col2im
  , col2vid
  , vid2col
  ) where

import qualified Data.Vector.Storable        as U (unsafeFromForeignPtr0,
                                                   unsafeToForeignPtr0)

import           Foreign                     (mallocForeignPtrArray, withForeignPtr)
import           Foreign.Ptr                 (Ptr)
import           Numeric.LinearAlgebra       (Matrix, cols, flatten, rows)
import qualified Numeric.LinearAlgebra.Devel as U
import           System.IO.Unsafe            (unsafePerformIO)

import           Grenade.Types

col2vid :: Int -> Int -> Int -> Int -> Int -> Int -> Matrix F -> Matrix F
col2vid kernelRows kernelColumns strideRows strideColumns height width dataCol =
  let channels = cols dataCol `div` (kernelRows * kernelColumns)
  in  col2im_c channels height width kernelRows kernelColumns strideRows strideColumns dataCol

col2im :: Int -> Int -> Int -> Int -> Int -> Int -> Matrix F -> Matrix F
col2im kernelRows kernelColumns strideRows strideColumns height width dataCol =
  let channels = 1
  in  col2im_c channels height width kernelRows kernelColumns strideRows strideColumns dataCol

col2im_c :: Int -> Int -> Int -> Int -> Int -> Int -> Int -> Matrix F -> Matrix F
col2im_c channels height width kernelRows kernelColumns strideRows strideColumns dataCol =
  let vec = flatten dataCol
  in unsafePerformIO $ do
    outPtr <- mallocForeignPtrArray (height * width * channels)
    let (inPtr, _) = U.unsafeToForeignPtr0 vec

    withForeignPtr inPtr $ \inPtr' ->
      withForeignPtr outPtr $ \outPtr' ->
        col2im_cpu inPtr' channels height width kernelRows kernelColumns strideRows strideColumns outPtr'

    let matVec = U.unsafeFromForeignPtr0 outPtr (height * width * channels)
    return $ U.matrixFromVector U.RowMajor (height * channels) width matVec

foreign import ccall unsafe
    col2im_cpu
      :: Ptr F -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Ptr F -> IO ()

vid2col :: Int -> Int -> Int -> Int -> Int -> Int -> Matrix F -> Matrix F
vid2col kernelRows kernelColumns strideRows strideColumns height width dataVid =
  let channels = rows dataVid `div` height
  in  im2col_c channels height width kernelRows kernelColumns strideRows strideColumns dataVid


im2col :: Int -> Int -> Int -> Int -> Matrix F -> Matrix F
im2col kernelRows kernelColumns strideRows strideColumns dataIm =
  let channels = 1
      height = rows dataIm
      width  = cols dataIm
  in  im2col_c channels height width kernelRows kernelColumns strideRows strideColumns dataIm

im2col_c :: Int -> Int -> Int -> Int -> Int -> Int -> Int -> Matrix F -> Matrix F
im2col_c channels height width kernelRows kernelColumns strideRows strideColumns dataIm =
  let vec    = flatten dataIm
      rowOut = (height - kernelRows) `div` strideRows + 1
      colOut = (width - kernelColumns) `div` strideColumns + 1
      kernelSize      = kernelRows * kernelColumns
      numberOfPatches = rowOut * colOut
  in unsafePerformIO $ do
    outPtr <- mallocForeignPtrArray (numberOfPatches * kernelSize * channels)
    let (inPtr, _) = U.unsafeToForeignPtr0 vec

    withForeignPtr inPtr $ \inPtr' ->
      withForeignPtr outPtr $ \outPtr' ->
        im2col_cpu inPtr' channels height width kernelRows kernelColumns strideRows strideColumns outPtr'

    let matVec = U.unsafeFromForeignPtr0 outPtr (numberOfPatches * kernelSize * channels)
    return $ U.matrixFromVector U.RowMajor numberOfPatches (kernelSize * channels) matVec

foreign import ccall unsafe
    im2col_cpu
      :: Ptr F -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Ptr F -> IO ()
