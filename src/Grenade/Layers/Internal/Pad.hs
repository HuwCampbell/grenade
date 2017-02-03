{-# LANGUAGE CPP                      #-}
{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE DataKinds                #-}
{-# LANGUAGE ScopedTypeVariables      #-}
{-# LANGUAGE GADTs                    #-}
{-# LANGUAGE TypeOperators            #-}
{-# LANGUAGE MultiParamTypeClasses    #-}
{-# LANGUAGE FlexibleContexts         #-}

#if __GLASGOW_HASKELL__ == 800
{-# OPTIONS_GHC -fno-warn-redundant-constraints #-}
#endif
module Grenade.Layers.Internal.Pad (
    pad
  , crop
  ) where

import           Data.Maybe ( fromJust )
import           Data.Proxy
import qualified Data.Vector.Storable as U ( unsafeToForeignPtr0, unsafeFromForeignPtr0 )

import           GHC.TypeLits

import           Grenade.Core

import           Foreign ( mallocForeignPtrArray, withForeignPtr )
import           Foreign.Ptr ( Ptr )

import           Numeric.LinearAlgebra ( flatten )
import           Numeric.LinearAlgebra.Static ( extract )

import           System.IO.Unsafe ( unsafePerformIO )

pad :: forall padLeft padTop padRight padBottom rows rows' cols cols' channels.
       ( KnownNat padLeft
       , KnownNat padTop
       , KnownNat padRight
       , KnownNat padBottom
       , KnownNat rows
       , KnownNat rows'
       , KnownNat cols
       , KnownNat cols'
       , KnownNat channels
       , rows' ~ (rows + padTop + padBottom)
       , cols' ~ (cols + padLeft + padRight)
       , KnownNat (rows' * channels)
       ) => Proxy padLeft
         -> Proxy padTop
         -> Proxy padRight
         -> Proxy padBottom
         -> S ('D3 rows cols channels)
         -> S ('D3 rows' cols' channels)
pad _ _ _ _ (S3D m) =
  let channels        = fromIntegral $ natVal (Proxy :: Proxy channels)
      padLeft         = fromIntegral $ natVal (Proxy :: Proxy padLeft)
      padTop          = fromIntegral $ natVal (Proxy :: Proxy padTop)
      padRight        = fromIntegral $ natVal (Proxy :: Proxy padRight)
      padBottom       = fromIntegral $ natVal (Proxy :: Proxy padBottom)
      rows            = fromIntegral $ natVal (Proxy :: Proxy rows)
      cols            = fromIntegral $ natVal (Proxy :: Proxy cols)
      rows'           = fromIntegral $ natVal (Proxy :: Proxy rows')
      cols'           = fromIntegral $ natVal (Proxy :: Proxy cols')
      outMatSize      = rows' * cols' * channels

      vec             = flatten (extract m)
  in unsafePerformIO $ do
    outPtr        <- mallocForeignPtrArray outMatSize
    let (inPtr, _) = U.unsafeToForeignPtr0 vec

    withForeignPtr inPtr $ \inPtr' ->
      withForeignPtr outPtr $ \outPtr' ->
        pad_cpu inPtr' channels rows cols padLeft padTop padRight padBottom outPtr'

    let matVec = U.unsafeFromForeignPtr0 outPtr outMatSize
    return (fromJust $ fromStorable matVec)

foreign import ccall unsafe
    pad_cpu
      :: Ptr Double -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Ptr Double -> IO ()


crop :: forall padLeft padTop padRight padBottom rows rows' cols cols' channels.
       ( KnownNat padLeft
       , KnownNat padTop
       , KnownNat padRight
       , KnownNat padBottom
       , KnownNat rows
       , KnownNat cols
       , KnownNat cols'
       , KnownNat channels
       , rows' ~ (rows + padTop + padBottom)
       , cols' ~ (cols + padLeft + padRight)
       , KnownNat (rows * channels)
       ) => Proxy padLeft
         -> Proxy padTop
         -> Proxy padRight
         -> Proxy padBottom
         -> S ('D3 rows' cols' channels)
         -> S ('D3 rows cols channels)
crop _ _ _ _ (S3D m) =
  let channels        = fromIntegral $ natVal (Proxy :: Proxy channels)
      padLeft         = fromIntegral $ natVal (Proxy :: Proxy padLeft)
      padTop          = fromIntegral $ natVal (Proxy :: Proxy padTop)
      padRight        = fromIntegral $ natVal (Proxy :: Proxy padRight)
      padBottom       = fromIntegral $ natVal (Proxy :: Proxy padBottom)
      rows            = fromIntegral $ natVal (Proxy :: Proxy rows)
      cols            = fromIntegral $ natVal (Proxy :: Proxy cols)
      outMatSize      = rows * cols * channels

      vec             = flatten (extract m)
  in unsafePerformIO $ do
    outPtr        <- mallocForeignPtrArray outMatSize
    let (inPtr, _) = U.unsafeToForeignPtr0 vec

    withForeignPtr inPtr $ \inPtr' ->
      withForeignPtr outPtr $ \outPtr' ->
        crop_cpu inPtr' channels rows cols padLeft padTop padRight padBottom outPtr'

    let matVec = U.unsafeFromForeignPtr0 outPtr outMatSize
    return (fromJust $ fromStorable matVec)

foreign import ccall unsafe
    crop_cpu
      :: Ptr Double -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Ptr Double -> IO ()
