{-# LANGUAGE DeriveGeneric #-}

module Grenade.Utils.Accuracy
    ( Accuracy
    , HyperParamAccuracy(..)
    , accuracyM
    ) where

import Grenade.Core.LearningParameters

import Grenade.Utils.Accuracy.Internal

import GHC.Generics

import Data.Validity

import Data.Aeson (ToJSON, FromJSON)

data HyperParamAccuracy = HyperParamAccuracy
    { hyperParam :: LearningParameters
    , testAccuracies :: [Accuracy]
    , validationAccuracies :: [Accuracy]
    , trainAccuracies :: [Accuracy]
    } deriving (Show, Eq, Generic)

instance ToJSON HyperParamAccuracy

instance FromJSON HyperParamAccuracy

instance Validity HyperParamAccuracy
