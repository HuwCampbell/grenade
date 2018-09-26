{-# LANGUAGE TemplateHaskell     #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE ConstraintKinds     #-}
{-# LANGUAGE TypeOperators       #-}

{-# OPTIONS_GHC -fno-warn-missing-signatures #-}
module Test.Grenade.Layers.Internal.Convolution.Accelerate where

import           Grenade.Layers.Internal.Convolution.Accelerate
import           Numeric.LinearAlgebra hiding (uniformSample, konst, (===))

import           Hedgehog
import qualified Hedgehog.Gen as Gen
import qualified Hedgehog.Range as Range

import qualified Test.Grenade.Layers.Internal.Reference as Reference
import           Test.Hedgehog.Compat

import           Data.Array.Accelerate.Interpreter
import qualified Data.Array.Accelerate as A
import           Data.Array.Accelerate (Z(..), (:.)(..))

prop_im2col_col2im_indexes_symmetrical_with_kernel_stride =
  let factors n = [x | x <- [1..n], n `mod` x == 0]
  in  property $ do
        height   <- forAll $ choose 2 100
        width    <- forAll $ choose 2 100
        kernel_h <- forAll $ (height `div`)    <$> Gen.element (factors height)
        kernel_w <- forAll $ (width `div`)     <$> Gen.element (factors width)

        let stride_h = kernel_h
            stride_w = kernel_w
            size = Z :. height :. width
            _imIx2colIx = imIx2colIx kernel_h kernel_w stride_h stride_w size
            _colIx2imIx = colIx2imIx kernel_h kernel_w stride_h stride_w size

        x <- forAll $ choose 0 (width - 1)
        y <- forAll $ choose 0 (height - 1)
        let input = Z :. y :. x

        let out      = (_colIx2imIx . _imIx2colIx) input
        input === out

prop_im2col_col2im_symmetrical_with_kernel_stride =
  let factors n = [x | x <- [1..n], n `mod` x == 0]
  in  property $ do
        height   <- forAll $ choose 2 100
        width    <- forAll $ choose 2 100
        kernel_h <- forAll $ (height `div`)    <$> Gen.element (factors height)
        kernel_w <- forAll $ (width `div`)     <$> Gen.element (factors width)
        let imageShape = (Z :. height :. width)
        input    <- forAll $ A.fromList imageShape <$> Gen.list (Range.singleton $ height * width) (Gen.realFloat $ Range.linearFracFrom 0 (-100) 100)

        let stride_h = kernel_h
            stride_w = kernel_w
            _col2im = col2im kernel_h kernel_w stride_h stride_w height width
            _im2col = im2col kernel_h kernel_w stride_h stride_w

        let out      = run1 (_col2im . _im2col) input
        input === out

prop_im2col_col2im_behaves_as_reference =
  let ok extent kernel = [stride | stride <- [1..extent], (extent - kernel) `mod` stride == 0]
  in  property $ do
        height   <- forAll $ choose 2 100
        width    <- forAll $ choose 2 100
        kernel_h <- forAll $ choose 2 (height - 1)
        kernel_w <- forAll $ choose 2 (width - 1)
        stride_h <- forAll $ Gen.element (ok height kernel_h)
        stride_w <- forAll $ Gen.element (ok width kernel_w)
        let contents = [0..]

        let inputRef = (height >< width) contents
            imageShape = (Z :. height :. width)
            input    =  A.fromList imageShape contents

            outFast       = run1 (im2col kernel_h kernel_w stride_h stride_w) input
            retFast       = run1 (col2im kernel_h kernel_w stride_h stride_w height width) outFast

            outReference  = Reference.im2col kernel_h kernel_w stride_h stride_w inputRef
            retReference  = Reference.col2im kernel_h kernel_w stride_h stride_w height width outReference

        (A.toList outFast) === (concat $ toLists outReference)
        (A.toList retFast) === (concat $ toLists retReference)

tests :: IO Bool
tests = checkParallel $$(discover)
