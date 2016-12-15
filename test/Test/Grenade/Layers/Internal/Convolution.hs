{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE GADTs             #-}
{-# LANGUAGE ScopedTypeVariables             #-}
{-# LANGUAGE ConstraintKinds         #-}
{-# LANGUAGE TypeOperators         #-}

{-# OPTIONS_GHC -fno-warn-missing-signatures #-}
module Test.Grenade.Layers.Internal.Convolution where

-- import           Control.Monad.Random

import           Grenade.Layers.Internal.Convolution

import           Numeric.LinearAlgebra hiding (uniformSample, konst, (===))

import           Disorder.Jack

import qualified Test.Grenade.Layers.Internal.Reference as Reference

prop_im2col_col2im_symmetrical_with_kernel_stride =
  let factors n = [x | x <- [1..n], n `mod` x == 0]
  in  gamble (choose (2, 100)) $ \height ->
        gamble (choose (2, 100)) $ \width ->
          gamble ((height `div`) <$> elements (factors height)) $ \kernel_h ->
            gamble ((width `div`) <$> elements (factors width)) $ \kernel_w ->
              gamble (listOfN (height * width) (height * width) sizedRealFrac) $ \input ->
                let input'   = (height >< width) input
                    stride_h = kernel_h
                    stride_w = kernel_w
                    out      = col2im kernel_h kernel_w stride_h stride_w height width . im2col kernel_h kernel_w stride_h stride_w $ input'
                in  input' === out

prop_im2col_col2im_behaves_as_reference =
  let ok extent kernel = [stride | stride <- [1..extent], (extent - kernel) `mod` stride == 0]
  in  gamble (choose (2, 100)) $ \height ->
        gamble (choose (2, 100)) $ \width ->
          gamble (choose (2, height - 1)) $ \kernel_h ->
            gamble (choose (2, width - 1)) $ \kernel_w ->
              gamble (elements (ok height kernel_h)) $ \stride_h ->
                gamble (elements (ok width kernel_w)) $ \stride_w ->
                  gamble (listOfN (height * width) (height * width) sizedRealFrac) $ \input ->
                    let input'        = (height >< width) input
                        outFast       = im2col kernel_h kernel_w stride_h stride_w input'
                        retFast       = col2im kernel_h kernel_w stride_h stride_w height width outFast

                        outReference  = Reference.im2col kernel_h kernel_w stride_h stride_w  input'
                        retReference  = Reference.col2im kernel_h kernel_w stride_h stride_w height width outReference
                    in  outFast === outReference .&&. retFast === retReference


return []
tests :: IO Bool
tests = $quickCheckAll
