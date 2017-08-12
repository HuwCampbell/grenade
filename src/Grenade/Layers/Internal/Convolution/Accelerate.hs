{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Grenade.Layers.Internal.Convolution.Accelerate where

import qualified Prelude as P
import Data.Array.Accelerate

type Matrix e = Array DIM2 e

transpose :: Elt e => Acc (Array DIM2 e) -> Acc (Array DIM2 e)
transpose mat =
  let swap = lift1 $ \(Z:.x:.y) -> Z:.y:.x :: Z :. Exp Int :. Exp Int
  in  backpermute (swap $ shape mat) swap mat


im2colShape :: (P.Num a, P.Integral a) => a -> a -> a -> a -> (Z :. a :. a) -> Z :. a :. a
im2colShape kRs kCs sRs sCs (Z :. height :. width) =
  let
    rowOut = (height - kRs) `div` sRs + 1
    colOut = (width - kCs) `div` sCs + 1
    kernelSize = (kRs * kCs)
    numPatches = rowOut * colOut
  in Z :. numPatches :. kernelSize

colIx2imIx
  :: (P.Num a, P.Integral a)
  => a -> a
  -> a -> a
  -> (Z :. a :. a)
  -> (Z :. a :. a)
  -> (Z :. a :. a)
colIx2imIx kRs kCs sRs sCs (Z :. height :. width) (Z :. y' :. x') =
  let
    kX = x' `mod` kCs
    kY = x' `div` kCs
    rowOut = (height - kRs) `div` sRs + 1
    colOut = (width - kCs) `div` sCs + 1
    sX = y' `mod` colOut
    sY = y' `div` colOut
  in Z :. (sY * sRs + kY) :. (sX * sCs + kX)

im2col :: Int -> Int -> Int -> Int -> Acc (Matrix Double) -> Acc (Matrix Double)
im2col kernelRows kernelColumns strideRows strideColumns dataIm = dataCol
  where
    imSh :: Exp DIM2
    imSh = shape dataIm

    colSh :: Exp DIM2
    colSh = lift1 (im2colShape (lift kernelRows) (lift kernelColumns) (lift strideRows) (lift strideColumns) :: Z :. Exp Int :. Exp Int -> Z :. Exp Int :. Exp Int) imSh

    mapIxs :: Exp DIM2 -> Exp DIM2 -> Exp DIM2
    mapIxs = lift2 $ colIx2imIx (lift kernelRows :: Exp Int) (lift kernelColumns) (lift strideRows) (lift strideColumns)

    dataCol :: Acc (Matrix Double)
    dataCol = backpermute colSh (mapIxs imSh) dataIm

imIx2colIx
  :: (P.Num a, P.Integral a)
  => a
  -> a
  -> a
  -> a
  -> Z :. a :. a
  -> Z :. a :. a
  -> Z :. a :. a
imIx2colIx kRs kCs sRs sCs (Z :. height :. width) (Z :. y :. x) =
  let
    rowOut = (height - kRs) `div` sRs + 1
    colOut = (width - kCs) `div` sCs + 1
    sX = P.min (x `div` sCs) (colOut - 1)
    kX = x - sX * sCs
    sY = P.min (y `div` sRs) (rowOut - 1)
    kY = y - sY * sRs
    x' = kY * kCs + kX
    y' = sY * colOut + sX
  in Z :. y' :. x'

col2im :: Int -> Int -> Int -> Int -> Int -> Int -> Acc (Matrix Double) -> Acc (Matrix Double)
col2im kernelRows kernelColumns strideRows strideColumns height width dataCol = dataIm
  where
    imSh :: DIM2
    imSh = Z :. height :. width

    mapIxs :: Exp DIM2 -> Exp DIM2 -> Exp DIM2
    mapIxs = lift2 $ colIx2imIx (lift kernelRows :: Exp Int) (lift kernelColumns) (lift strideRows) (lift strideColumns)

    dataIm :: Acc (Matrix Double)
    dataIm = permute (+) (fill (constant imSh) 0) (mapIxs $ lift imSh) dataCol
