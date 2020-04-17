{-# LANGUAGE ForeignFunctionInterface #-}
module Grenade.Layers.Internal.Pad (
    pad
  , crop
  ) where

import qualified Data.Vector.Storable        as U (unsafeFromForeignPtr0,
                                                   unsafeToForeignPtr0)

import           Foreign                     (mallocForeignPtrArray, withForeignPtr)
import           Foreign.Ptr                 (Ptr)
import           Numeric.LinearAlgebra       (Matrix, flatten)
import qualified Numeric.LinearAlgebra.Devel as U
import           System.IO.Unsafe            (unsafePerformIO)

import           Grenade.Types

pad :: Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Matrix F -> Matrix F
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
      :: Ptr F -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Ptr F -> IO ()

crop :: Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Matrix F -> Matrix F
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
      :: Ptr F -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Ptr F -> IO ()
