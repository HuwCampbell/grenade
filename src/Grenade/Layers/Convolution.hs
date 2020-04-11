{-# LANGUAGE CPP                   #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE UndecidableInstances  #-}
{-|
Module      : Grenade.Layers.Convolution
Description : Convolution layer
Copyright   : (c) Huw Campbell, 2016-2017
License     : BSD2
Stability   : experimental

This module provides the Convolution layer, which is critical in many computer vision tasks.

-}
module Grenade.Layers.Convolution (
    Convolution (..)
  , Convolution' (..)
  , randomConvolution
  ) where

import           Control.Monad.Random hiding ( fromList )
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

import           Numeric.LinearAlgebra hiding ( uniformSample, konst )
import qualified Numeric.LinearAlgebra as LA
import           Numeric.LinearAlgebra.Static hiding ((|||), build, toRows)

import           Grenade.Core
import           Grenade.Layers.Internal.Convolution
import           Grenade.Layers.Internal.Update

-- | A convolution layer for a neural network.
--   This uses the im2col convolution trick popularised by Caffe, which essentially turns the
--   many, many, many, many loop convolution into a single matrix multiplication.
--
--   The convolution layer takes all of the kernels for the convolution, which are flattened
--   and then put into columns in the matrix.
--
--   The kernel size dictates which input and output sizes will "fit". Fitting the equation:
--   `out = (in - kernel) / stride + 1` for both dimensions.
--
--   One probably shouldn't build their own layer, but rather use the randomConvolution function.
data Convolution :: Nat -- Number of channels, for the first layer this could be RGB for instance.
                 -> Nat -- Number of filters, this is the number of channels output by the layer.
                 -> Nat -- The number of rows in the kernel filter
                 -> Nat -- The number of column in the kernel filter
                 -> Nat -- The row stride of the convolution filter
                 -> Nat -- The columns stride of the convolution filter
                 -> Type where
  Convolution :: ( KnownNat channels
                 , KnownNat filters
                 , KnownNat kernelRows
                 , KnownNat kernelColumns
                 , KnownNat strideRows
                 , KnownNat strideColumns
                 , KnownNat kernelFlattened
                 , kernelFlattened ~ (kernelRows * kernelColumns * channels))
              => !(L kernelFlattened filters) -- The kernel filter weights
              -> !(L kernelFlattened filters) -- The last kernel update (or momentum)
              -> Convolution channels filters kernelRows kernelColumns strideRows strideColumns

data Convolution' :: Nat -- Number of channels, for the first layer this could be RGB for instance.
                  -> Nat -- Number of filters, this is the number of channels output by the layer.
                  -> Nat -- The number of rows in the kernel filter
                  -> Nat -- The number of column in the kernel filter
                  -> Nat -- The row stride of the convolution filter
                  -> Nat -- The columns stride of the convolution filter
                  -> Type where
  Convolution' :: ( KnownNat channels
                  , KnownNat filters
                  , KnownNat kernelRows
                  , KnownNat kernelColumns
                  , KnownNat strideRows
                  , KnownNat strideColumns
                  , KnownNat kernelFlattened
                  , kernelFlattened ~ (kernelRows * kernelColumns * channels))
               => !(L kernelFlattened filters) -- The kernel filter gradient
               -> Convolution' channels filters kernelRows kernelColumns strideRows strideColumns

instance Show (Convolution c f k k' s s') where
  show (Convolution a _) = renderConv a
    where
      renderConv mm =
        let m  = extract mm
            ky = fromIntegral $ natVal (Proxy :: Proxy k)
            rs = LA.toColumns m
            ms = map (take ky) $ toLists . reshape ky <$> rs

            render n'  | n' <= 0.2  = ' '
                       | n' <= 0.4  = '.'
                       | n' <= 0.6  = '-'
                       | n' <= 0.8  = '='
                       | otherwise =  '#'

            px = (fmap . fmap . fmap) render ms
        in unlines $ foldl1 (zipWith (\a' b' -> a' ++ "   |   " ++ b')) $ px

randomConvolution :: ( MonadRandom m
                     , KnownNat channels
                     , KnownNat filters
                     , KnownNat kernelRows
                     , KnownNat kernelColumns
                     , KnownNat strideRows
                     , KnownNat strideColumns
                     , KnownNat kernelFlattened
                     , kernelFlattened ~ (kernelRows * kernelColumns * channels))
                  => m (Convolution channels filters kernelRows kernelColumns strideRows strideColumns)
randomConvolution = do
    s     <- getRandom
    let wN = uniformSample s (-1) 1
        mm = konst 0
    return $ Convolution wN mm

instance ( KnownNat channels
         , KnownNat filters
         , KnownNat kernelRows
         , KnownNat kernelColumns
         , KnownNat strideRows
         , KnownNat strideColumns
         , KnownNat (kernelRows * kernelColumns * channels)
         ) => UpdateLayer (Convolution channels filters kernelRows kernelColumns strideRows strideColumns) where
  type Gradient (Convolution channels filters kernelRows kernelColumns strideRows strideColumns) = (Convolution' channels filters kernelRows kernelColumns strideRows strideColumns)
  runUpdate lp (Convolution oldKernel oldMomentum) (Convolution' kernelGradient) =
    let (newKernel, newMomentum) = descendMatrix (learningRate lp) (learningMomentum lp) (learningRegulariser lp) oldKernel kernelGradient oldMomentum
    in Convolution newKernel newMomentum

  createRandom = randomConvolution

instance ( KnownNat channels
         , KnownNat filters
         , KnownNat kernelRows
         , KnownNat kernelColumns
         , KnownNat strideRows
         , KnownNat strideColumns
         , KnownNat (kernelRows * kernelColumns * channels)
         ) => Serialize (Convolution channels filters kernelRows kernelColumns strideRows strideColumns) where
  put (Convolution w _) = putListOf put . toList . flatten . extract $ w
  get = do
      let f  = fromIntegral $ natVal (Proxy :: Proxy filters)
      wN    <- maybe (fail "Vector of incorrect size") return . create . reshape f . LA.fromList =<< getListOf get
      let mm = konst 0
      return $ Convolution wN mm

-- | A three dimensional image (or 2d with many channels) can have
--   an appropriately sized convolution filter run across it.
instance ( KnownNat kernelRows
         , KnownNat kernelCols
         , KnownNat filters
         , KnownNat strideRows
         , KnownNat strideCols
         , KnownNat inputRows
         , KnownNat inputCols
         , KnownNat outputRows
         , KnownNat outputCols
         , KnownNat channels
         , ((outputRows - 1) * strideRows) ~ (inputRows - kernelRows)
         , ((outputCols - 1) * strideCols) ~ (inputCols - kernelCols)
         , KnownNat (kernelRows * kernelCols * channels)
         , KnownNat (outputRows * filters)
         ) => Layer (Convolution channels filters kernelRows kernelCols strideRows strideCols) ('D3 inputRows inputCols channels) ('D3 outputRows outputCols filters) where

  type Tape (Convolution channels filters kernelRows kernelCols strideRows strideCols) ('D3 inputRows inputCols channels) ('D3 outputRows outputCols filters) = S ('D3 inputRows inputCols channels)

  runForwards (Convolution kernel _) (S3D input) =
    let ex = extract input
        ek = extract kernel
        ix = fromIntegral $ natVal (Proxy :: Proxy inputRows)
        iy = fromIntegral $ natVal (Proxy :: Proxy inputCols)
        kx = fromIntegral $ natVal (Proxy :: Proxy kernelRows)
        ky = fromIntegral $ natVal (Proxy :: Proxy kernelCols)
        sx = fromIntegral $ natVal (Proxy :: Proxy strideRows)
        sy = fromIntegral $ natVal (Proxy :: Proxy strideCols)
        ox = fromIntegral $ natVal (Proxy :: Proxy outputRows)
        oy = fromIntegral $ natVal (Proxy :: Proxy outputCols)

        c  = vid2col kx ky sx sy ix iy ex
        mt = c LA.<> ek
        r  = col2vid 1 1 1 1 ox oy mt
        rs = fromJust . create $ r
    in  (S3D input, S3D rs)
  runBackwards (Convolution kernel _) (S3D input) (S3D dEdy) =
    let ex = extract input
        ix = fromIntegral $ natVal (Proxy :: Proxy inputRows)
        iy = fromIntegral $ natVal (Proxy :: Proxy inputCols)
        kx = fromIntegral $ natVal (Proxy :: Proxy kernelRows)
        ky = fromIntegral $ natVal (Proxy :: Proxy kernelCols)
        sx = fromIntegral $ natVal (Proxy :: Proxy strideRows)
        sy = fromIntegral $ natVal (Proxy :: Proxy strideCols)
        ox = fromIntegral $ natVal (Proxy :: Proxy outputRows)
        oy = fromIntegral $ natVal (Proxy :: Proxy outputCols)

        c  = vid2col kx ky sx sy ix iy ex

        eo = extract dEdy
        ek = extract kernel

        vs = vid2col 1 1 1 1 ox oy eo

        kN = fromJust . create $ tr c LA.<> vs

        dW = vs LA.<> tr ek

        xW = col2vid kx ky sx sy ix iy dW
    in  (Convolution' kN, S3D . fromJust . create $ xW)


-- | A two dimentional image may have a convolution filter applied to it
instance ( KnownNat kernelRows
         , KnownNat kernelCols
         , KnownNat filters
         , KnownNat strideRows
         , KnownNat strideCols
         , KnownNat inputRows
         , KnownNat inputCols
         , KnownNat outputRows
         , KnownNat outputCols
         , ((outputRows - 1) * strideRows) ~ (inputRows - kernelRows)
         , ((outputCols - 1) * strideCols) ~ (inputCols - kernelCols)
         , KnownNat (kernelRows * kernelCols * 1)
         , KnownNat (outputRows * filters)
         ) => Layer (Convolution 1 filters kernelRows kernelCols strideRows strideCols) ('D2 inputRows inputCols) ('D3 outputRows outputCols filters) where
  type Tape (Convolution 1 filters kernelRows kernelCols strideRows strideCols) ('D2 inputRows inputCols) ('D3 outputRows outputCols filters) = S ('D3 inputRows inputCols 1)
  runForwards c (S2D input) =
    runForwards c (S3D input :: S ('D3 inputRows inputCols 1))

  runBackwards c tape grads =
    case runBackwards c tape grads of
      (c', S3D back :: S ('D3 inputRows inputCols 1)) ->  (c', S2D back)


-- | A two dimensional image may have a convolution filter applied to it producing
--   a two dimensional image if both channels and filters is 1.
instance ( KnownNat kernelRows
         , KnownNat kernelCols
         , KnownNat strideRows
         , KnownNat strideCols
         , KnownNat inputRows
         , KnownNat inputCols
         , KnownNat outputRows
         , KnownNat outputCols
         , ((outputRows - 1) * strideRows) ~ (inputRows - kernelRows)
         , ((outputCols - 1) * strideCols) ~ (inputCols - kernelCols)
         , KnownNat (kernelRows * kernelCols * 1)
         , KnownNat (outputRows * 1)
         ) => Layer (Convolution 1 1 kernelRows kernelCols strideRows strideCols) ('D2 inputRows inputCols) ('D2 outputRows outputCols) where
  type Tape (Convolution 1 1 kernelRows kernelCols strideRows strideCols) ('D2 inputRows inputCols) ('D2 outputRows outputCols) = S ('D3 inputRows inputCols 1)
  runForwards c (S2D input) =
    case runForwards c (S3D input :: S ('D3 inputRows inputCols 1)) of
      (tps, S3D back :: S ('D3 outputRows outputCols 1)) ->  (tps, S2D back)

  runBackwards c tape (S2D grads) =
    case runBackwards c tape (S3D grads :: S ('D3 outputRows outputCols 1)) of
      (c', S3D back :: S ('D3 inputRows inputCols 1)) -> (c', S2D back)

-- | A three dimensional image can produce a 2D image from a convolution with 1 filter
instance ( KnownNat kernelRows
         , KnownNat kernelCols
         , KnownNat strideRows
         , KnownNat strideCols
         , KnownNat inputRows
         , KnownNat inputCols
         , KnownNat outputRows
         , KnownNat outputCols
         , KnownNat channels
         , ((outputRows - 1) * strideRows) ~ (inputRows - kernelRows)
         , ((outputCols - 1) * strideCols) ~ (inputCols - kernelCols)
         , KnownNat (kernelRows * kernelCols * channels)
         , KnownNat (outputRows * 1)
         ) => Layer (Convolution channels 1 kernelRows kernelCols strideRows strideCols) ('D3 inputRows inputCols channels) ('D2 outputRows outputCols) where
  type Tape (Convolution channels 1 kernelRows kernelCols strideRows strideCols) ('D3 inputRows inputCols channels) ('D2 outputRows outputCols) = S ('D3 inputRows inputCols channels)
  runForwards c input =
    case runForwards c input of
      (tps, S3D back :: S ('D3 outputRows outputCols 1)) ->  (tps, S2D back)

  runBackwards c tape (S2D grads) =
    runBackwards c tape (S3D grads :: S ('D3 outputRows outputCols 1))
