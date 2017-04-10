{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE CPP                 #-}
{-# LANGUAGE TemplateHaskell     #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE KindSignatures      #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE LambdaCase          #-}
{-# OPTIONS_GHC -fno-warn-missing-signatures #-}
module Test.Grenade.Layers.Nonlinear where

import           Data.Singletons

#if __GLASGOW_HASKELL__ < 800
import           Data.Proxy
#endif

import           Grenade
import           GHC.TypeLits

import           Hedgehog

import           Test.Jack.Compat
import           Test.Jack.Hmatrix
import           Test.Jack.TypeLits

import           Numeric.LinearAlgebra.Static ( norm_Inf )

prop_sigmoid_grad :: Property
prop_sigmoid_grad = property $
    blindForAll genShape >>= \case
        (SomeSing (r :: Sing s)) ->
            withSingI r $
                blindForAll genOfShape >>= \(ds :: S s) ->
                    let (tape, f  :: S s)  = runForwards Logit ds
                        ((), ret  :: S s)  = runBackwards Logit tape (1 :: S s)
                        (_, numer :: S s)  = runForwards Logit (ds + 0.0001)
                        numericalGradient  = (numer - f) * 10000
                    in assert ((case numericalGradient - ret of
                           (S1D x) -> norm_Inf x < 0.0001
                           (S2D x) -> norm_Inf x < 0.0001
                           (S3D x) -> norm_Inf x < 0.0001) :: Bool)

prop_tanh_grad :: Property
prop_tanh_grad = property $
    blindForAll genShape >>= \case
        (SomeSing (r :: Sing s)) ->
            withSingI r $
                blindForAll genOfShape >>=  \(ds :: S s) ->
                    let (tape, f  :: S s)  = runForwards Tanh ds
                        ((), ret  :: S s)  = runBackwards Tanh tape (1 :: S s)
                        (_, numer :: S s)  = runForwards Tanh (ds + 0.0001)
                        numericalGradient  = (numer - f) * 10000
                    in assert ((case numericalGradient - ret of
                           (S1D x) -> norm_Inf x < 0.001
                           (S2D x) -> norm_Inf x < 0.001
                           (S3D x) -> norm_Inf x < 0.001) :: Bool)

prop_softmax_grad :: Property
prop_softmax_grad = property $
    blindForAll genNat >>= \case
        (SomeNat (_ :: Proxy s)) ->
            blindForAll genOfShape >>= \(ds :: S ('D1 s)) ->
                let (tape, f  :: S ('D1 s))  = runForwards Relu ds
                    ((), ret  :: S ('D1 s))  = runBackwards Relu tape (1 :: S ('D1 s))
                    (_, numer :: S ('D1 s))  = runForwards Relu (ds + 0.0001)
                    numericalGradient  = (numer - f) * 10000
                in assert ((case numericalGradient - ret of
                        (S1D x) -> norm_Inf x < 0.0001) :: Bool)



tests :: IO Bool
tests = $$(checkConcurrent)

