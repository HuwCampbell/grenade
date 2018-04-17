{-# LANGUAGE DeriveGeneric #-}

module Grenade.Utils.Accuracy.Internal where

import Data.Validity

import Control.Monad.Catch

import Data.Aeson (ToJSON, FromJSON)

import GHC.Generics

newtype Accuracy = Accuracy Double deriving (Show, Eq, Generic)

data AccuracyNotInRange = AccuracyNotInRange deriving (Show, Eq)

instance Exception AccuracyNotInRange where
    displayException AccuracyNotInRange = "The accuracy is not in [0,1]."

accuracyM :: MonadThrow m => Double -> m Accuracy
accuracyM x = case 0 <= x && x <= 1 of
    False -> throwM AccuracyNotInRange
    True -> pure $ Accuracy x

instance ToJSON Accuracy

instance FromJSON Accuracy

instance Validity Accuracy where
    validate (Accuracy x) = 0 <= x && x <= 1 <?@> "The accuracy is in [0,1]"
