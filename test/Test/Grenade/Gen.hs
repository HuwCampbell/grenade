{-# OPTIONS_GHC -fno-warn-orphans #-}

module Test.Grenade.Gen where

import Grenade.Core.LearningParameters

import Data.GenValidity

import Grenade.Utils.Accuracy
import Grenade.Utils.Accuracy.Internal

import Test.QuickCheck (choose, listOf)
import Test.QuickCheck.Gen (suchThat)

instance GenUnchecked LearningParameters

instance GenValid LearningParameters where
    genValid = do
        rate <- genValid `suchThat` (> 0)
        momentum <- genValid `suchThat` (>= 0)
        LearningParameters rate momentum <$> genValid `suchThat` (>= 0)

instance GenUnchecked Accuracy

instance GenValid Accuracy where
    genValid = Accuracy <$> choose (0,1)

instance GenUnchecked HyperParamAccuracy

instance GenValid HyperParamAccuracy where
    genValid = do
        param <- genValid
        testAcc <- listOf genValid
        validationAcc <- listOf genValid
        HyperParamAccuracy param testAcc validationAcc <$> listOf genValid
