{-# LANGUAGE ForeignFunctionInterface #-}
module Grenade.Layers.Internal.Pooling (
    poolForward
  , poolBackward
  ) where

import qualified Data.Vector.Storable as U ( unsafeToForeignPtr0, unsafeFromForeignPtr0 )

import           Foreign ( mallocForeignPtrArray, withForeignPtr )
import           Foreign.Ptr ( Ptr )

import           Numeric.LinearAlgebra ( Matrix , flatten )
import qualified Numeric.LinearAlgebra.Devel as U

import           System.IO.Unsafe ( unsafePerformIO )

poolForward :: Int -> Int -> Int -> Int -> Int -> Int -> Int -> Matrix Double -> Matrix Double
poolForward channels height width kernelRows kernelColumns strideRows strideColumns dataIm =
  let vec             = flatten dataIm
      rowOut          = (height - kernelRows) `div` strideRows + 1
      colOut          = (width - kernelColumns) `div` strideColumns + 1
      numberOfPatches = rowOut * colOut
  in unsafePerformIO $ do
    outPtr <- mallocForeignPtrArray (numberOfPatches * channels)
    let (inPtr, _) = U.unsafeToForeignPtr0 vec

    withForeignPtr inPtr $ \inPtr' ->
      withForeignPtr outPtr $ \outPtr' ->
        pool_forwards_cpu inPtr' channels height width kernelRows kernelColumns strideRows strideColumns outPtr'

    let matVec = U.unsafeFromForeignPtr0 outPtr (numberOfPatches * channels)
    return $ U.matrixFromVector U.RowMajor (rowOut * channels) colOut matVec

foreign import ccall unsafe
    pool_forwards_cpu
      :: Ptr Double -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Ptr Double -> IO ()

poolBackward :: Int -> Int -> Int -> Int -> Int -> Int -> Int -> Matrix Double -> Matrix Double -> Matrix Double
poolBackward channels height width kernelRows kernelColumns strideRows strideColumns dataIm dataGrad =
  let vecIm     = flatten dataIm
      vecGrad   = flatten dataGrad
  in unsafePerformIO $ do
    outPtr <- mallocForeignPtrArray (height * width * channels)
    let (imPtr, _) = U.unsafeToForeignPtr0 vecIm
    let (gradPtr, _) = U.unsafeToForeignPtr0 vecGrad

    withForeignPtr imPtr $ \imPtr' ->
      withForeignPtr gradPtr $ \gradPtr' ->
        withForeignPtr outPtr $ \outPtr' ->
          pool_backwards_cpu imPtr' gradPtr' channels height width kernelRows kernelColumns strideRows strideColumns outPtr'

    let matVec = U.unsafeFromForeignPtr0 outPtr (height * width * channels)
    return $ U.matrixFromVector U.RowMajor (height * channels) width matVec

foreign import ccall unsafe
    pool_backwards_cpu
      :: Ptr Double -> Ptr Double -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Ptr Double -> IO ()
