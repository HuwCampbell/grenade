{-# LANGUAGE TemplateHaskell     #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE ConstraintKinds     #-}
{-# LANGUAGE TypeOperators       #-}

{-# OPTIONS_GHC -fno-warn-missing-signatures #-}
module Test.Grenade.Layers.Internal.Pooling where

import           Grenade.Layers.Internal.Pooling

import           Numeric.LinearAlgebra hiding (uniformSample, konst, (===))

import           Hedgehog
import qualified Hedgehog.Gen as Gen
import qualified Hedgehog.Range as Range

import qualified Test.Grenade.Layers.Internal.Reference as Reference
import           Test.Hedgehog.Compat

prop_poolForwards_poolBackwards_behaves_as_reference =
  let ok extent kernel = [stride | stride <- [1..extent], (extent - kernel) `mod` stride == 0]
      output extent kernel stride = (extent - kernel) `div` stride + 1
  in  property $ do
        height   <- forAll $ choose 2 100
        width    <- forAll $ choose 2 100
        kernel_h <- forAll $ choose 1 (height - 1)
        kernel_w <- forAll $ choose 1 (width - 1)
        stride_h <- forAll $ Gen.element (ok height kernel_h)
        stride_w <- forAll $ Gen.element (ok width kernel_w)
        input    <- forAll $ (height >< width) <$> Gen.list (Range.singleton $ height * width) (Gen.realFloat $ Range.linearFracFrom 0 (-100) 100)

        let outFast       = poolForward 1 height width kernel_h kernel_w stride_h stride_w input
        let retFast       = poolBackward 1 height width kernel_h kernel_w stride_h stride_w input outFast

        let outReference  = Reference.poolForward kernel_h kernel_w stride_h stride_w (output height kernel_h stride_h) (output width kernel_w stride_w) input
        let retReference  = Reference.poolBackward kernel_h kernel_w stride_h stride_w  input outReference

        outFast === outReference
        retFast === retReference


tests :: IO Bool
tests = checkParallel $$(discover)
