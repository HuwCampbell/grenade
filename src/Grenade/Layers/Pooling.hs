{-# LANGUAGE CPP                   #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE StandaloneDeriving    #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE UndecidableInstances  #-}
{-|
Module      : Grenade.Core.Pooling
Description : Max Pooling layer for 2D and 3D images
Copyright   : (c) Huw Campbell, 2016-2017
License     : BSD2
Stability   : experimental
-}
module Grenade.Layers.Pooling (
    Pooling (..)
  ) where

import           Data.Maybe
import           Data.Proxy
import           Data.Serialize
import           Data.Singletons.TypeLits

#if MIN_VERSION_base(4,11,0)
import           GHC.TypeLits hiding (natVal)
#else
import           GHC.TypeLits
#endif
#if MIN_VERSION_base(4,9,0)
import           Data.Kind (Type)
#endif

import           Grenade.Core
import           Grenade.Layers.Internal.Pooling

import           Numeric.LinearAlgebra.Static as LAS hiding ((|||), build, toRows)

-- | A pooling layer for a neural network.
--
--   Does a max pooling, looking over a kernel similarly to the convolution network, but returning
--   maxarg only. This layer is often used to provide minor amounts of translational invariance.
--
--   The kernel size dictates which input and output sizes will "fit". Fitting the equation:
--   `out = (in - kernel) / stride + 1` for both dimensions.
--
data Pooling :: Nat -> Nat -> Nat -> Nat -> Type where
  Pooling :: Pooling kernelRows kernelColumns strideRows strideColumns

instance Show (Pooling k k' s s') where
  show Pooling = "Pooling"

instance UpdateLayer (Pooling kernelRows kernelColumns strideRows strideColumns) where
  type Gradient (Pooling kernelRows kernelColumns strideRows strideColumns) = ()
  runUpdate _ Pooling _ = Pooling
  createRandom = return Pooling

instance Serialize (Pooling kernelRows kernelColumns strideRows strideColumns) where
  put _ = return ()
  get = return Pooling

-- | A two dimentional image can be pooled.
instance ( KnownNat kernelRows
         , KnownNat kernelColumns
         , KnownNat strideRows
         , KnownNat strideColumns
         , KnownNat inputRows
         , KnownNat inputColumns
         , KnownNat outputRows
         , KnownNat outputColumns
         , ((outputRows - 1) * strideRows) ~ (inputRows - kernelRows)
         , ((outputColumns - 1) * strideColumns) ~ (inputColumns - kernelColumns)
         ) => Layer (Pooling kernelRows kernelColumns strideRows strideColumns) ('D2 inputRows inputColumns) ('D2 outputRows outputColumns) where
  type Tape (Pooling kernelRows kernelColumns strideRows strideColumns) ('D2 inputRows inputColumns) ('D2 outputRows outputColumns) = S ('D2 inputRows inputColumns)
  runForwards Pooling (S2D input) =
    let height = fromIntegral $ natVal (Proxy :: Proxy inputRows)
        width  = fromIntegral $ natVal (Proxy :: Proxy inputColumns)
        kx = fromIntegral $ natVal (Proxy :: Proxy kernelRows)
        ky = fromIntegral $ natVal (Proxy :: Proxy kernelColumns)
        sx = fromIntegral $ natVal (Proxy :: Proxy strideRows)
        sy = fromIntegral $ natVal (Proxy :: Proxy strideColumns)
        ex = extract input
        r  = poolForward 1 height width kx ky sx sy ex
        rs = fromJust . create $ r
    in  (S2D input, S2D rs)
  runBackwards Pooling (S2D input) (S2D dEdy) =
    let height = fromIntegral $ natVal (Proxy :: Proxy inputRows)
        width  = fromIntegral $ natVal (Proxy :: Proxy inputColumns)
        kx = fromIntegral $ natVal (Proxy :: Proxy kernelRows)
        ky = fromIntegral $ natVal (Proxy :: Proxy kernelColumns)
        sx = fromIntegral $ natVal (Proxy :: Proxy strideRows)
        sy = fromIntegral $ natVal (Proxy :: Proxy strideColumns)
        ex = extract input
        eo = extract dEdy
        vs = poolBackward 1 height width kx ky sx sy ex eo
    in  ((), S2D . fromJust . create $ vs)


-- | A three dimensional image can be pooled on each layer.
instance ( KnownNat kernelRows
         , KnownNat kernelColumns
         , KnownNat strideRows
         , KnownNat strideColumns
         , KnownNat inputRows
         , KnownNat inputColumns
         , KnownNat outputRows
         , KnownNat outputColumns
         , KnownNat channels
         , KnownNat (outputRows * channels)
         , ((outputRows - 1) * strideRows) ~ (inputRows - kernelRows)
         , ((outputColumns - 1) * strideColumns) ~ (inputColumns - kernelColumns)
         ) => Layer (Pooling kernelRows kernelColumns strideRows strideColumns) ('D3 inputRows inputColumns channels) ('D3 outputRows outputColumns channels) where
  type Tape (Pooling kernelRows kernelColumns strideRows strideColumns) ('D3 inputRows inputColumns channels) ('D3 outputRows outputColumns channels) = S ('D3 inputRows inputColumns channels)
  runForwards Pooling (S3D input) =
    let ix = fromIntegral $ natVal (Proxy :: Proxy inputRows)
        iy = fromIntegral $ natVal (Proxy :: Proxy inputColumns)
        kx = fromIntegral $ natVal (Proxy :: Proxy kernelRows)
        ky = fromIntegral $ natVal (Proxy :: Proxy kernelColumns)
        sx = fromIntegral $ natVal (Proxy :: Proxy strideRows)
        sy = fromIntegral $ natVal (Proxy :: Proxy strideColumns)
        ch = fromIntegral $ natVal (Proxy :: Proxy channels)
        ex = extract input
        r  = poolForward ch ix iy kx ky sx sy ex
        rs = fromJust . create $ r
    in  (S3D input, S3D rs)
  runBackwards Pooling (S3D input) (S3D dEdy) =
    let ix = fromIntegral $ natVal (Proxy :: Proxy inputRows)
        iy = fromIntegral $ natVal (Proxy :: Proxy inputColumns)
        kx = fromIntegral $ natVal (Proxy :: Proxy kernelRows)
        ky = fromIntegral $ natVal (Proxy :: Proxy kernelColumns)
        sx = fromIntegral $ natVal (Proxy :: Proxy strideRows)
        sy = fromIntegral $ natVal (Proxy :: Proxy strideColumns)
        ch = fromIntegral $ natVal (Proxy :: Proxy channels)
        ex = extract input
        eo = extract dEdy
        vs = poolBackward ch ix iy kx ky sx sy ex eo
    in  ((), S3D . fromJust . create $ vs)
