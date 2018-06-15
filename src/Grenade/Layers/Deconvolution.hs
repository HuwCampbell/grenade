{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE RecordWildCards       #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE FlexibleContexts      #-}
{-|
Module      : Grenade.Layers.Deconvolution
Description : Deconvolution layer
Copyright   : (c) Huw Campbell, 2016-2017
License     : BSD2
Stability   : experimental

A deconvolution layer is in many ways a convolution layer in reverse.
It learns a kernel to apply to each pixel location, spreading it out
into a larger layer.

This layer is important for image generation tasks, such as GANs on
images.
-}
module Grenade.Layers.Deconvolution (
    Deconvolution (..)
  , Deconvolution' (..)
  ) where

import           Data.Maybe
import           Data.Proxy
import           Data.Serialize
import           Data.Singletons.TypeLits

import           GHC.TypeLits
import Control.DeepSeq (NFData (..))


import           Numeric.LinearAlgebra hiding ( uniformSample, konst )
import qualified Numeric.LinearAlgebra as LA
import           Numeric.LinearAlgebra.Static hiding ((|||), build, toRows)

import           Grenade.Core
import           Grenade.Layers.Internal.Convolution
import           Grenade.Layers.Internal.Update

-- | A Deconvolution layer for a neural network.
--   This uses the im2col Convolution trick popularised by Caffe.
--
--   The Deconvolution layer is a way of spreading out a single response
--   into a larger image, and is useful in generating images.
--
data Deconvolution :: Nat -- Number of channels, for the first layer this could be RGB for instance.
                   -> Nat -- Number of filters, this is the number of channels output by the layer.
                   -> Nat -- The number of rows in the kernel filter
                   -> Nat -- The number of column in the kernel filter
                   -> Nat -- The row stride of the Deconvolution filter
                   -> Nat -- The columns stride of the Deconvolution filter
                   -> * where
  Deconvolution :: ( KnownNat channels
                   , KnownNat filters
                   , KnownNat kernelRows
                   , KnownNat kernelColumns
                   , KnownNat strideRows
                   , KnownNat strideColumns
                   , KnownNat kernelFlattened
                   , kernelFlattened ~ (kernelRows * kernelColumns * filters))
                 => !(L kernelFlattened channels) -- The kernel filter weights
                 -> !(L kernelFlattened channels) -- The last kernel update (or momentum)
                 -> Deconvolution channels filters kernelRows kernelColumns strideRows strideColumns

instance NFData (Deconvolution c f k k' s s') where 
  rnf (Deconvolution a b) = rnf a `seq` rnf b `seq` ()


data Deconvolution' :: Nat -- Number of channels, for the first layer this could be RGB for instance.
                    -> Nat -- Number of filters, this is the number of channels output by the layer.
                    -> Nat -- The number of rows in the kernel filter
                    -> Nat -- The number of column in the kernel filter
                    -> Nat -- The row stride of the Deconvolution filter
                    -> Nat -- The columns stride of the Deconvolution filter
                    -> * where
  Deconvolution' :: ( KnownNat channels
                  , KnownNat filters
                  , KnownNat kernelRows
                  , KnownNat kernelColumns
                  , KnownNat strideRows
                  , KnownNat strideColumns
                  , KnownNat kernelFlattened
                  , kernelFlattened ~ (kernelRows * kernelColumns * filters))
               => !(L kernelFlattened channels) -- The kernel filter gradient
               -> Deconvolution' channels filters kernelRows kernelColumns strideRows strideColumns

instance NFData (Deconvolution' c f k k' s s') where 
  rnf (Deconvolution' a) = rnf a `seq` ()


instance Show (Deconvolution c f k k' s s') where
  show (Deconvolution a _) = renderConv a
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

instance ( KnownNat c
         , KnownNat f
         , KnownNat k
         , KnownNat k'
         , KnownNat s
         , KnownNat s'
         , KnownNat ((k * k') * f)
         , KnownNat ((k * k') * c)) => RandomLayer (Deconvolution c f k k' s s') where
  createRandomWith m = do
    wN <- getRandomMatrix i i m
    let mm = konst 0
    return $ Deconvolution wN mm
    where i = natVal (Proxy :: Proxy ((k * k') * c))

instance ( KnownNat channels
         , KnownNat filters
         , KnownNat kernelRows
         , KnownNat kernelColumns
         , KnownNat strideRows
         , KnownNat strideColumns
         , KnownNat (kernelRows * kernelColumns * filters)
         ) => UpdateLayer (Deconvolution channels filters kernelRows kernelColumns strideRows strideColumns) where
  type Gradient (Deconvolution channels filters kernelRows kernelColumns strideRows strideColumns) = (Deconvolution' channels filters kernelRows kernelColumns strideRows strideColumns)
  runUpdate LearningParameters {..} (Deconvolution oldKernel oldMomentum) (Deconvolution' kernelGradient) =
    let (newKernel, newMomentum) = descendMatrix learningRate learningMomentum learningRegulariser oldKernel kernelGradient oldMomentum
    in Deconvolution newKernel newMomentum


instance ( KnownNat channels
         , KnownNat filters
         , KnownNat kernelRows
         , KnownNat kernelColumns
         , KnownNat strideRows
         , KnownNat strideColumns
         , KnownNat (kernelRows * kernelColumns * filters)
         ) => Serialize (Deconvolution channels filters kernelRows kernelColumns strideRows strideColumns) where
  put (Deconvolution w _) = putListOf put . toList . flatten . extract $ w
  get = do
      let f  = fromIntegral $ natVal (Proxy :: Proxy channels)
      wN    <- maybe (fail "Vector of incorrect size") return . create . reshape f . LA.fromList =<< getListOf get
      let mm = konst 0
      return $ Deconvolution wN mm

-- | A two dimentional image may have a Deconvolution filter applied to it
instance ( KnownNat kernelRows
         , KnownNat kernelCols
         , KnownNat filters
         , KnownNat strideRows
         , KnownNat strideCols
         , KnownNat inputRows
         , KnownNat inputCols
         , KnownNat outputRows
         , KnownNat outputCols
         , ((inputRows - 1) * strideRows) ~ (outputRows - kernelRows)
         , ((inputCols - 1) * strideCols) ~ (outputCols - kernelCols)
         , KnownNat (kernelRows * kernelCols * filters)
         , KnownNat (outputRows * filters)
         ) => Layer (Deconvolution 1 filters kernelRows kernelCols strideRows strideCols) ('D2 inputRows inputCols) ('D3 outputRows outputCols filters) where
  type Tape (Deconvolution 1 filters kernelRows kernelCols strideRows strideCols) ('D2 inputRows inputCols) ('D3 outputRows outputCols filters) = S ('D3 inputRows inputCols 1)
  runForwards c (S2D input) =
    runForwards c (S3D input :: S ('D3 inputRows inputCols 1))

  runBackwards c tape grads =
    case runBackwards c tape grads of
      (c', S3D back :: S ('D3 inputRows inputCols 1)) ->  (c', S2D back)

-- | A two dimentional image may have a Deconvolution filter applied to it
instance ( KnownNat kernelRows
         , KnownNat kernelCols
         , KnownNat strideRows
         , KnownNat strideCols
         , KnownNat inputRows
         , KnownNat inputCols
         , KnownNat outputRows
         , KnownNat outputCols
         , ((inputRows - 1) * strideRows) ~ (outputRows - kernelRows)
         , ((inputCols - 1) * strideCols) ~ (outputCols - kernelCols)
         , KnownNat (kernelRows * kernelCols * 1)
         , KnownNat (outputRows * 1)
         ) => Layer (Deconvolution 1 1 kernelRows kernelCols strideRows strideCols) ('D2 inputRows inputCols) ('D2 outputRows outputCols) where
  type Tape (Deconvolution 1 1 kernelRows kernelCols strideRows strideCols) ('D2 inputRows inputCols) ('D2 outputRows outputCols) = S ('D3 inputRows inputCols 1)
  runForwards c (S2D input) =
    case runForwards c (S3D input :: S ('D3 inputRows inputCols 1)) of
      (tps, S3D fore :: S ('D3 outputRows outputCols 1)) ->  (tps, S2D fore)

  runBackwards c tape (S2D grads) =
    case runBackwards c tape (S3D grads :: S ('D3 outputRows outputCols 1)) of
      (c', S3D back :: S ('D3 inputRows inputCols 1)) ->  (c', S2D back)

-- | A two dimentional image may have a Deconvolution filter applied to it
instance ( KnownNat kernelRows
         , KnownNat kernelCols
         , KnownNat strideRows
         , KnownNat strideCols
         , KnownNat inputRows
         , KnownNat inputCols
         , KnownNat outputRows
         , KnownNat outputCols
         , ((inputRows - 1) * strideRows) ~ (outputRows - kernelRows)
         , ((inputCols - 1) * strideCols) ~ (outputCols - kernelCols)
         , KnownNat (kernelRows * kernelCols * 1)
         , KnownNat (outputRows * 1)
         , KnownNat channels
         ) => Layer (Deconvolution channels 1 kernelRows kernelCols strideRows strideCols) ('D3 inputRows inputCols channels) ('D2 outputRows outputCols) where
  type Tape (Deconvolution channels 1 kernelRows kernelCols strideRows strideCols) ('D3 inputRows inputCols channels) ('D2 outputRows outputCols) = S ('D3 inputRows inputCols channels)
  runForwards c input =
    case runForwards c input of
      (tps, S3D fore :: S ('D3 outputRows outputCols 1)) ->  (tps, S2D fore)

  runBackwards c tape (S2D grads) =
    runBackwards c tape (S3D grads :: S ('D3 outputRows outputCols 1))

-- | A three dimensional image (or 2d with many channels) can have
--   an appropriately sized Deconvolution filter run across it.
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
         , ((inputRows - 1) * strideRows) ~ (outputRows - kernelRows)
         , ((inputCols - 1) * strideCols) ~ (outputCols - kernelCols)
         , KnownNat (kernelRows * kernelCols * filters)
         , KnownNat (outputRows * filters)
         ) => Layer (Deconvolution channels filters kernelRows kernelCols strideRows strideCols) ('D3 inputRows inputCols channels) ('D3 outputRows outputCols filters) where

  type Tape (Deconvolution channels filters kernelRows kernelCols strideRows strideCols) ('D3 inputRows inputCols channels) ('D3 outputRows outputCols filters) = S ('D3 inputRows inputCols channels)

  runForwards (Deconvolution kernel _) (S3D input) =
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

        c  = vid2col 1 1 1 1 ix iy ex

        mt = c LA.<> tr ek

        r  = col2vid kx ky sx sy ox oy mt
        rs = fromJust . create $ r
    in  (S3D input, S3D rs)
  runBackwards (Deconvolution kernel _) (S3D input) (S3D dEdy) =
    let ex = extract input
        ix = fromIntegral $ natVal (Proxy :: Proxy inputRows)
        iy = fromIntegral $ natVal (Proxy :: Proxy inputCols)
        kx = fromIntegral $ natVal (Proxy :: Proxy kernelRows)
        ky = fromIntegral $ natVal (Proxy :: Proxy kernelCols)
        sx = fromIntegral $ natVal (Proxy :: Proxy strideRows)
        sy = fromIntegral $ natVal (Proxy :: Proxy strideCols)
        ox = fromIntegral $ natVal (Proxy :: Proxy outputRows)
        oy = fromIntegral $ natVal (Proxy :: Proxy outputCols)

        c  = vid2col 1 1 1 1 ix iy ex

        eo = extract dEdy
        ek = extract kernel

        vs = vid2col kx ky sx sy ox oy eo

        kN = fromJust . create . tr $ tr c LA.<> vs

        dW = vs LA.<> ek

        xW = col2vid 1 1 1 1 ix iy dW
    in  (Deconvolution' kN, S3D . fromJust . create $ xW)
