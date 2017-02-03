{-# LANGUAGE ForeignFunctionInterface #-}
module Grenade.Layers.Internal.Pad (
    pad
  , crop
  ) where

import qualified Data.Vector.Storable as U ( unsafeToForeignPtr0, unsafeFromForeignPtr0 )

import           Foreign ( mallocForeignPtrArray, withForeignPtr )
import           Foreign.Ptr ( Ptr )

import           Numeric.LinearAlgebra ( flatten, Matrix )
import qualified Numeric.LinearAlgebra.Devel as U

import           System.IO.Unsafe ( unsafePerformIO )

pad :: Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Matrix Double -> Matrix Double
pad channels padLeft padTop padRight padBottom rows cols rows' cols' m
 = let outMatSize      = rows' * cols' * channels
       vec             = flatten m
   in unsafePerformIO $ do
     outPtr        <- mallocForeignPtrArray outMatSize
     let (inPtr, _) = U.unsafeToForeignPtr0 vec

     withForeignPtr inPtr $ \inPtr' ->
       withForeignPtr outPtr $ \outPtr' ->
         pad_cpu inPtr' channels rows cols padLeft padTop padRight padBottom outPtr'

     let matVec = U.unsafeFromForeignPtr0 outPtr outMatSize
     return (U.matrixFromVector U.RowMajor (rows' * channels) cols' matVec)
{-# INLINE pad #-}

foreign import ccall unsafe
    pad_cpu
      :: Ptr Double -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Ptr Double -> IO ()

crop :: Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Matrix Double -> Matrix Double
crop channels padLeft padTop padRight padBottom rows cols _ _ m
 = let outMatSize      = rows * cols * channels
       vec             = flatten m
   in unsafePerformIO $ do
     outPtr        <- mallocForeignPtrArray outMatSize
     let (inPtr, _) = U.unsafeToForeignPtr0 vec

     withForeignPtr inPtr $ \inPtr' ->
       withForeignPtr outPtr $ \outPtr' ->
         crop_cpu inPtr' channels rows cols padLeft padTop padRight padBottom outPtr'

     let matVec = U.unsafeFromForeignPtr0 outPtr outMatSize
     return (U.matrixFromVector U.RowMajor (rows * channels) cols matVec)

foreign import ccall unsafe
    crop_cpu
      :: Ptr Double -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Ptr Double -> IO ()
