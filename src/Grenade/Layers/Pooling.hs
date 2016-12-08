{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE StandaloneDeriving    #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE PolyKinds             #-}

module Grenade.Layers.Pooling (
    Pooling (..)
  ) where

import           Data.Maybe
import           Data.Proxy
import           Data.Singletons.TypeLits
import           GHC.TypeLits

import           Grenade.Core.Network
import           Grenade.Core.Shape
import           Grenade.Core.Vector
import           Grenade.Layers.Internal.Pooling

import           Numeric.LinearAlgebra.Static as LAS hiding ((|||), build, toRows)

-- | A pooling layer for a neural network.
--   Does a max pooling, looking over a kernel similarly to the convolution network, but returning
--   maxarg only. This layer is often used to provide minor amounts of translational invariance.
--
--   The kernel size dictates which input and output sizes will "fit". Fitting the equation:
--   `out = (in - kernel) / stride + 1` for both dimensions.
--
data Pooling :: Nat -> Nat -> Nat -> Nat -> * where
  Pooling :: Pooling kernelRows kernelColumns strideRows strideColumns

instance Show (Pooling k k' s s') where
  show Pooling = "Pooling"

instance UpdateLayer (Pooling kernelRows kernelColumns strideRows strideColumns) where
  type Gradient (Pooling kr kc sr sc) = ()
  runUpdate _ Pooling _ = Pooling
  createRandom = return Pooling

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
  runForwards Pooling (S2D' input) =
    let kx = fromIntegral $ natVal (Proxy :: Proxy kernelRows)
        ky = fromIntegral $ natVal (Proxy :: Proxy kernelColumns)
        sx = fromIntegral $ natVal (Proxy :: Proxy strideRows)
        sy = fromIntegral $ natVal (Proxy :: Proxy strideColumns)
        ox = fromIntegral $ natVal (Proxy :: Proxy outputRows)
        oy = fromIntegral $ natVal (Proxy :: Proxy outputColumns)
        ex = extract input
        r  = poolForward kx ky sx sy ox oy $ ex
        rs = fromJust . create $ r
    in  S2D' $ rs
  runBackwards Pooling (S2D' input) (S2D' dEdy) =
    let kx = fromIntegral $ natVal (Proxy :: Proxy kernelRows)
        ky = fromIntegral $ natVal (Proxy :: Proxy kernelColumns)
        sx = fromIntegral $ natVal (Proxy :: Proxy strideRows)
        sy = fromIntegral $ natVal (Proxy :: Proxy strideColumns)
        ex = extract input
        eo = extract dEdy
        vs = poolBackward kx ky sx sy ex eo
    in  ((), S2D' . fromJust . create $ vs)


-- | A three dimensional image can be pooled on each layer.
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
         ) => Layer (Pooling kernelRows kernelColumns strideRows strideColumns) ('D3 inputRows inputColumns channels) ('D3 outputRows outputColumns channels) where
  runForwards Pooling (S3D' input) =
    let ix = fromIntegral $ natVal (Proxy :: Proxy inputRows)
        iy = fromIntegral $ natVal (Proxy :: Proxy inputColumns)
        kx = fromIntegral $ natVal (Proxy :: Proxy kernelRows)
        ky = fromIntegral $ natVal (Proxy :: Proxy kernelColumns)
        sx = fromIntegral $ natVal (Proxy :: Proxy strideRows)
        sy = fromIntegral $ natVal (Proxy :: Proxy strideColumns)
        ox = fromIntegral $ natVal (Proxy :: Proxy outputRows)
        oy = fromIntegral $ natVal (Proxy :: Proxy outputColumns)
        ex = fmap extract input
        r  = poolForwardList kx ky sx sy ix iy ox oy ex
        rs = fmap (fromJust . create) r
    in  S3D' rs
  runBackwards Pooling (S3D' input) (S3D' dEdy) =
    let ix = fromIntegral $ natVal (Proxy :: Proxy inputRows)
        iy = fromIntegral $ natVal (Proxy :: Proxy inputColumns)
        kx = fromIntegral $ natVal (Proxy :: Proxy kernelRows)
        ky = fromIntegral $ natVal (Proxy :: Proxy kernelColumns)
        sx = fromIntegral $ natVal (Proxy :: Proxy strideRows)
        sy = fromIntegral $ natVal (Proxy :: Proxy strideColumns)
        ex = fmap extract input
        eo = fmap extract dEdy
        ez = vectorZip (,) ex eo
        vs = poolBackwardList kx ky sx sy ix iy ez
    in  ((), S3D' . fmap (fromJust . create) $ vs)
