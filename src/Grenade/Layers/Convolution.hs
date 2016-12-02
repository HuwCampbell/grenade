{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE StandaloneDeriving    #-}
{-# LANGUAGE RecordWildCards       #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE PolyKinds             #-}
{-# LANGUAGE PatternGuards         #-}

module Grenade.Layers.Convolution (
    Convolution (..)
  , randomConvolution
  , im2col
  , vid2col
  , col2im
  , col2vid
  , fittingStarts
  ) where

import           Control.Monad.Random hiding (fromList)
import           Data.Maybe
import           Data.Proxy
import           Data.Singletons.TypeLits
import           GHC.TypeLits

import           Numeric.LinearAlgebra hiding (uniformSample, konst)
import qualified Numeric.LinearAlgebra as LA
import           Numeric.LinearAlgebra.Static hiding ((|||), build, toRows)

import           Grenade.Core.Network
import           Grenade.Core.Shape
import           Grenade.Core.Vector

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
data Convolution :: Nat -- ^ Number of channels, for the first layer this could be RGB for instance.
                 -> Nat -- ^ Number of filters, this is the number of channels output by the layer.
                 -> Nat -- ^ The number of rows in the kernel filter
                 -> Nat -- ^ The number of column in the kernel filter
                 -> Nat -- ^ The row stride of the convolution filter
                 -> Nat -- ^ The columns stride of the convolution filter
                 -> * where
  Convolution :: ( KnownNat channels
                 , KnownNat filters
                 , KnownNat kernelRows
                 , KnownNat kernelColumns
                 , KnownNat strideRows
                 , KnownNat strideColumns
                 , KnownNat kernelFlattened
                 , kernelFlattened ~ (kernelRows * kernelColumns * channels))
              => !(L kernelFlattened filters) -- ^ The kernel filter weights
              -> !(L kernelFlattened filters) -- ^ The last kernel update (or momentum)
              -> Convolution channels filters kernelRows kernelColumns strideRows strideColumns

data Convolution' :: Nat -- ^ Number of channels, for the first layer this could be RGB for instance.
                  -> Nat -- ^ Number of filters, this is the number of channels output by the layer.
                  -> Nat -- ^ The number of rows in the kernel filter
                  -> Nat -- ^ The number of column in the kernel filter
                  -> Nat -- ^ The row stride of the convolution filter
                  -> Nat -- ^ The columns stride of the convolution filter
                  -> * where
  Convolution' :: ( KnownNat channels
                  , KnownNat filters
                  , KnownNat kernelRows
                  , KnownNat kernelColumns
                  , KnownNat strideRows
                  , KnownNat strideColumns
                  , KnownNat kernelFlattened
                  , kernelFlattened ~ (kernelRows * kernelColumns * channels))
               => !(L kernelFlattened filters) -- ^ The kernel filter gradient
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
    s  :: Int <- getRandom
    let wN = uniformSample s (-1) 1
        mm = konst 0
    return $ Convolution wN mm

instance ( Monad m
         , KnownNat kernelRows
         , KnownNat kernelCols
         , KnownNat channels
         , KnownNat filters
         , KnownNat strideRows
         , KnownNat strideCols
         , kernelFlattened ~ (kernelRows * kernelColumns * channels)
         ) => UpdateLayer m (Convolution channels filters kernelRows kernelCols strideRows strideCols) where
  type Gradient (Convolution channels filters kernelRows kernelCols strideRows strideCols) = (Convolution' channels filters kernelRows kernelCols strideRows strideCols)
  runUpdate LearningParameters {..} (Convolution oldKernel oldMomentum) (Convolution' kernelGradient) = do
    let newMomentum    = konst learningMomentum * oldMomentum - konst learningRate * kernelGradient
        regulariser    = konst (learningRegulariser * learningRate) * oldKernel
        newKernel      = oldKernel + newMomentum - regulariser
    return $ Convolution newKernel newMomentum

-- | A two dimentional image may have a convolution filter applied to it
instance ( Monad m
         , KnownNat kernelRows
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
         ) => Layer m (Convolution 1 filters kernelRows kernelCols strideRows strideCols) ('D2 inputRows inputCols) ('D3 outputRows outputCols filters) where
  runForwards (Convolution kernel _) (S2D' input) =
    let ex = extract input
        ek = extract kernel
        kx = fromIntegral $ natVal (Proxy :: Proxy kernelRows)
        ky = fromIntegral $ natVal (Proxy :: Proxy kernelCols)
        sx = fromIntegral $ natVal (Proxy :: Proxy strideRows)
        sy = fromIntegral $ natVal (Proxy :: Proxy strideCols)
        ox = fromIntegral $ natVal (Proxy :: Proxy outputRows)
        oy = fromIntegral $ natVal (Proxy :: Proxy outputCols)
        c  = im2col kx ky sx sy ex
        mt = c LA.<> ek
        r  = col2vid 1 1 1 1 ox oy mt
        rs = fmap (fromJust . create) r
    in  return . S3D' $ mkVector rs
  runBackards (Convolution kernel _) (S2D' input) (S3D' dEdy) =
    let ex = extract input
        ix = fromIntegral $ natVal (Proxy :: Proxy inputRows)
        iy = fromIntegral $ natVal (Proxy :: Proxy inputCols)
        kx = fromIntegral $ natVal (Proxy :: Proxy kernelRows)
        ky = fromIntegral $ natVal (Proxy :: Proxy kernelCols)
        sx = fromIntegral $ natVal (Proxy :: Proxy strideRows)
        sy = fromIntegral $ natVal (Proxy :: Proxy strideCols)
        ox = fromIntegral $ natVal (Proxy :: Proxy outputRows)
        oy = fromIntegral $ natVal (Proxy :: Proxy outputCols)
        c  = im2col kx ky sx sy ex

        eo = vecToList $ fmap extract dEdy
        ek = extract kernel

        vs = vid2col 1 1 1 1 ox oy eo

        kN = fromJust . create $ tr c LA.<> vs
        dW = vs LA.<> tr ek

        xW = col2im kx ky sx sy ix iy dW
    in  return (Convolution' kN, S2D' . fromJust . create $ xW)


-- | A three dimensional image (or 2d with many channels) can have
--   an appropriately sized convolution filter run across it.
instance ( Monad m
         , KnownNat kernelRows
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
         ) => Layer m (Convolution channels filters kernelRows kernelCols strideRows strideCols) ('D3 inputRows inputCols channels) ('D3 outputRows outputCols filters) where
  runForwards (Convolution kernel _) (S3D' input) =
    let ex = vecToList $ fmap extract input
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
        rs = fmap (fromJust . create) r
    in  return . S3D' $ mkVector rs
  runBackards (Convolution kernel _) (S3D' input) (S3D' dEdy) =
    let ex = vecToList $ fmap extract input
        ix = fromIntegral $ natVal (Proxy :: Proxy inputRows)
        iy = fromIntegral $ natVal (Proxy :: Proxy inputCols)
        kx = fromIntegral $ natVal (Proxy :: Proxy kernelRows)
        ky = fromIntegral $ natVal (Proxy :: Proxy kernelCols)
        sx = fromIntegral $ natVal (Proxy :: Proxy strideRows)
        sy = fromIntegral $ natVal (Proxy :: Proxy strideCols)
        ox = fromIntegral $ natVal (Proxy :: Proxy outputRows)
        oy = fromIntegral $ natVal (Proxy :: Proxy outputCols)
        c  = vid2col kx ky sx sy ix iy ex

        eo = vecToList $ fmap extract dEdy
        ek = extract kernel

        vs = vid2col 1 1 1 1 ox oy eo

        kN = fromJust . create $ tr c LA.<> vs

        dW = vs LA.<> tr ek

        xW = col2vid kx ky sx sy ix iy dW
    in  return (Convolution' kN, S3D' . mkVector . fmap (fromJust . create) $ xW)

im2col :: Int -> Int -> Int -> Int -> Matrix Double -> Matrix Double
im2col nrows ncols srows scols m =
  let starts = fittingStarts (rows m) nrows srows (cols m) ncols scols
  in  im2colFit starts nrows ncols m

im2colFit :: [(Int,Int)] -> Int -> Int -> Matrix Double -> Matrix Double
im2colFit starts nrows ncols m =
  let imRows = fmap (\start -> flatten $ subMatrix start (nrows, ncols) m) starts
  in  fromRows imRows

vid2col :: Int -> Int -> Int -> Int -> Int -> Int -> [Matrix Double] -> Matrix Double
vid2col nrows ncols srows scols inputrows inputcols ms =
  let starts = fittingStarts inputrows nrows srows inputcols ncols scols
      subs   = fmap (im2colFit starts nrows ncols) ms
  in  foldl1 (|||) subs

col2vid :: Int -> Int -> Int -> Int -> Int -> Int -> Matrix Double -> [Matrix Double]
col2vid nrows ncols srows scols drows dcols m =
  let starts = fittingStart (cols m) (nrows * ncols) (nrows * ncols)
      r      = rows m
      mats   = fmap (\s -> subMatrix (0,s) (r, nrows * ncols) m) starts
      colSts = fittingStarts drows nrows srows dcols ncols scols
  in  fmap (col2imfit colSts nrows ncols drows dcols) mats

col2im :: Int -> Int -> Int -> Int -> Int -> Int -> Matrix Double -> Matrix Double
col2im krows kcols srows scols drows dcols m =
  let starts     = fittingStarts drows krows srows dcols kcols scols
  in  col2imfit starts krows kcols drows dcols m

col2imfit :: [(Int,Int)] -> Int -> Int -> Int -> Int -> Matrix Double -> Matrix Double
col2imfit starts krows kcols drows dcols m =
  let indicies   = fmap (\[a,b] -> (a,b)) $ sequence [[0..(krows-1)], [0..(kcols-1)]]
      convs      = fmap (zip indicies . toList) . toRows $ m
      pairs      = zip convs starts
      accums     = concat $ fmap (\(conv',(stx',sty')) -> fmap (\((ix,iy), val) -> ((ix + stx', iy + sty'), val)) conv') pairs
  in  accum (LA.konst 0 (drows, dcols)) (+) accums


-- | These functions are not even remotely safe, but it's only called from the statically typed
--   commands, so we should be good ?!?!?
--   Returns the starting sub matrix locations which fit inside the larger matrix for the
--   convolution. Takes into account the stride and kernel size.
fittingStarts :: Int -> Int -> Int -> Int -> Int -> Int -> [(Int,Int)]
fittingStarts nrows kernelrows steprows ncols kernelcols stepcolsh =
  let rs = fittingStart nrows kernelrows steprows
      cs = fittingStart ncols kernelcols stepcolsh
      ls = sequence [rs, cs]
  in  fmap (\[a,b] -> (a,b)) ls

-- | Returns the starting sub vector which fit inside the larger vector for the
--   convolution. Takes into account the stride and kernel size.
fittingStart :: Int -> Int -> Int -> [Int]
fittingStart width kernel steps =
  let go left | left + kernel < width
              = left : go (left + steps)
              | left + kernel == width
              = left : []
              | otherwise
              = error "Kernel and step do not fit in matrix."
  in  go 0
