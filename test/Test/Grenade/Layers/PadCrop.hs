{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE CPP                 #-}
{-# LANGUAGE TemplateHaskell     #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE KindSignatures      #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# OPTIONS_GHC -fno-warn-missing-signatures #-}

module Test.Grenade.Layers.PadCrop where

import           Grenade

import           Hedgehog

import           Numeric.LinearAlgebra.Static ( norm_Inf )

import           Test.Hedgehog.Hmatrix

prop_pad_crop :: Property
prop_pad_crop =
  let net :: Network '[Pad 2 3 4 6, Crop 2 3 4 6] '[ 'D3 7 9 5, 'D3 16 15 5, 'D3 7 9 5 ]
      net = Pad :~> Crop :~> NNil
  in  property $
    forAll genOfShape >>= \(d :: S ('D3 7 9 5)) ->
      let (tapes, res)  = runForwards  net d
          (_    , grad) = runBackwards net tapes d
      in  do assert $ d ~~~ res
             assert $ grad ~~~ d

prop_pad_crop_2d :: Property
prop_pad_crop_2d =
  let net :: Network '[Pad 2 3 4 6, Crop 2 3 4 6] '[ 'D2 7 9, 'D2 16 15, 'D2 7 9 ]
      net = Pad :~> Crop :~> NNil
  in  property $
    forAll genOfShape >>= \(d :: S ('D2 7 9)) ->
      let (tapes, res)  = runForwards  net d
          (_    , grad) = runBackwards net tapes d
      in  do assert $ d ~~~ res
             assert $ grad ~~~ d

(~~~) :: S x -> S x -> Bool
(S1D x) ~~~ (S1D y) = norm_Inf (x - y) < 0.00001
(S2D x) ~~~ (S2D y) = norm_Inf (x - y) < 0.00001
(S3D x) ~~~ (S3D y) = norm_Inf (x - y) < 0.00001


tests :: IO Bool
tests = checkParallel $$(discover)
