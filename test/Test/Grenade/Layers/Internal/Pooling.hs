{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE DataKinds       #-}
{-# LANGUAGE GADTs           #-}
{-# LANGUAGE ScopedTypeVariables             #-}
{-# LANGUAGE ConstraintKinds         #-}
{-# LANGUAGE TypeOperators         #-}

{-# OPTIONS_GHC -fno-warn-missing-signatures #-}
module Test.Grenade.Layers.Internal.Pooling where

import           Grenade.Layers.Internal.Pooling

import           Numeric.LinearAlgebra hiding (uniformSample, konst, (===))

import           Disorder.Jack

import qualified Test.Grenade.Layers.Internal.Reference as Reference

prop_poolForwards_poolBackwards_behaves_as_reference =
  let ok extent kernel = [stride | stride <- [1..extent], (extent - kernel) `mod` stride == 0]
      output extent kernel stride = (extent - kernel) `div` stride + 1
  in  gamble (choose (2, 100)) $ \height ->
        gamble (choose (2, 100)) $ \width ->
          gamble (choose (2, height - 1)) $ \kernel_h ->
            gamble (choose (2, width - 1)) $ \kernel_w ->
              gamble (elements (ok height kernel_h)) $ \stride_h ->
                gamble (elements (ok width kernel_w)) $ \stride_w ->
                  gamble (listOfN (height * width) (height * width) sizedRealFrac) $ \input ->
                    let input'        = (height >< width) input
                        outFast       = poolForward 1 height width kernel_h kernel_w stride_h stride_w input'
                        -- retFast       = poolBackward 1 height width kernel_h kernel_w stride_h stride_w input' outFast

                        outReference  = Reference.poolForward kernel_h kernel_w stride_h stride_w (output height kernel_h stride_h) (output width kernel_w stride_w) input'
                        -- retReference  = Reference.poolBackward kernel_h kernel_w stride_h stride_w  input' outReference
                    in  outFast === outReference -- .&&. retFast === retReference


prop_poolForwards_poolBackwards_symmetry =
  let factors n = [x | x <- [1..n], n `mod` x == 0]
      output extent kernel stride = (extent - kernel) `div` stride + 1
  in  gamble (choose (2, 100)) $ \height ->
        gamble (choose (2, 100)) $ \width ->
          gamble ((height `div`) <$> elements (factors height)) $ \kernel_h ->
            gamble ((width `div`) <$> elements (factors width)) $ \kernel_w ->
              gamble (listOfN (height * width) (height * width) sizedRealFrac) $ \input ->
                let input'        = (height >< width) input
                    stride_h      = kernel_h
                    stride_w      = kernel_w
                    outFast       = poolForward 1 height width kernel_h kernel_w stride_h stride_w input'
                    retFast       = poolBackward 1 height width kernel_h kernel_w stride_h stride_w input' outFast

                    outReference  = Reference.poolForward kernel_h kernel_w stride_h stride_w (output height kernel_h stride_h) (output width kernel_w stride_w) input'
                    retReference  = Reference.poolBackward kernel_h kernel_w stride_h stride_w  input' outReference
                in  outFast === outReference .&&. retFast === retReference


return []
tests :: IO Bool
tests = $quickCheckAll
