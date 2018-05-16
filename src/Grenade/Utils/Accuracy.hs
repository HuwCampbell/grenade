{-# LANGUAGE DeriveGeneric #-}

module Grenade.Utils.Accuracy
    ( ClassificationAccuracy
    , showClassificationAccuracy
    , constructProperFraction
    ) where

import Grenade.Utils.ProperFraction

-- | Classification accuracy of a network on a dataset.
type ClassificationAccuracy = ProperFraction

showClassificationAccuracy :: String -> ClassificationAccuracy -> String
showClassificationAccuracy name acc =
    "The " ++
    name ++ " accuracy is " ++ show (nOfPrecent * properToDouble acc) ++ "%."

nOfPrecent :: Double
nOfPrecent = 100
