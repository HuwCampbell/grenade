{-# LANGUAGE TemplateHaskell     #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE ConstraintKinds     #-}
{-# LANGUAGE TypeOperators       #-}

{-# OPTIONS_GHC -fno-warn-missing-signatures #-}
module Test.Grenade.Layers.Internal.Convolution where

import           Grenade.Layers.Internal.Convolution

import           Numeric.LinearAlgebra hiding (uniformSample, konst, (===))

import           Hedgehog
import qualified Hedgehog.Gen as Gen
import qualified Hedgehog.Range as Range

import qualified Test.Grenade.Layers.Internal.Reference as Reference
import           Test.Hedgehog.Compat

prop_im2col_col2im_symmetrical_with_kernel_stride =
  let factors n = [x | x <- [1..n], n `mod` x == 0]
  in  property $ do
        height   <- forAll $ choose 2 100
        width    <- forAll $ choose 2 100
        kernel_h <- forAll $ (height `div`)    <$> Gen.element (factors height)
        kernel_w <- forAll $ (width `div`)     <$> Gen.element (factors width)
        input    <- forAll $ (height >< width) <$> Gen.list (Range.singleton $ height * width) (Gen.realFloat $ Range.linearFracFrom 0 (-100) 100)

        let stride_h = kernel_h
        let stride_w = kernel_w
        let out      = col2im kernel_h kernel_w stride_h stride_w height width . im2col kernel_h kernel_w stride_h stride_w $ input
        input === out

prop_im2col_col2im_behaves_as_reference =
  let ok extent kernel = [stride | stride <- [1..extent], (extent - kernel) `mod` stride == 0]
  in  property $ do
        height   <- forAll (choose 2 100)
        width    <- forAll (choose 2 100)
        kernel_h <- forAll (choose 2 (height - 1))
        kernel_w <- forAll (choose 2 (width - 1))
        stride_h <- forAll (Gen.element (ok height kernel_h))
        stride_w <- forAll (Gen.element (ok width kernel_w))
        input    <- forAll ((height >< width) <$> Gen.list (Range.singleton $ height * width) (Gen.realFloat $ Range.linearFracFrom 0 (-100) 100))

        let outFast       = im2col kernel_h kernel_w stride_h stride_w input
        let retFast       = col2im kernel_h kernel_w stride_h stride_w height width outFast

        let outReference  = Reference.im2col kernel_h kernel_w stride_h stride_w  input
        let retReference  = Reference.col2im kernel_h kernel_w stride_h stride_w height width outReference

        outFast === outReference
        retFast === retReference

tests :: IO Bool
tests = checkParallel $$(discover)
