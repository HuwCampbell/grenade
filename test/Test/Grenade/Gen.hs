{-# OPTIONS_GHC -fno-warn-orphans #-}

module Test.Grenade.Gen where

import Grenade.Core.LearningParameters

import Data.GenValidity

import Test.QuickCheck.Gen (suchThat)

instance GenUnchecked LearningParameters

instance GenValid LearningParameters where
    genValid = do
        rate <- genValid `suchThat` (> 0)
        momentum <- genValid `suchThat` (>= 0)
        LearningParameters rate momentum <$> genValid `suchThat` (>= 0)
