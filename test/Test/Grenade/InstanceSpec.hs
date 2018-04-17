{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeApplications #-}

module Test.Grenade.InstanceSpec where

import Test.Hspec

import Test.Grenade.Gen ()

import Grenade.Core.LearningParameters

import Test.Validity
import Test.Validity.Aeson

tests :: IO Bool
tests = hspec spec >> pure True

spec :: Spec
spec = do
    genValidSpec @LearningParameters
    jsonSpecOnValid @LearningParameters
