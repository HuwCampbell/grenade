{-# LANGUAGE DeriveGeneric #-}

module Grenade.Utils.ProperFraction
    ( ProperFraction
    , constructProperFraction
    , properToDouble
    ) where

import Data.Aeson (FromJSON, ToJSON)
import Data.Validity

import GHC.Generics

newtype ProperFraction =
    ProperFraction Double
    deriving (Show, Eq, Generic, Ord)

instance ToJSON ProperFraction

instance FromJSON ProperFraction

instance Validity ProperFraction where
    validate (ProperFraction x) =
        mconcat
            [ delve "A proper fraction contains a valid Double" x
            , declare "A proper fraction is positive" $ x >= 0
            , declare "A proper fraction is smaller than 1" $ x <= 1
            ]

constructProperFraction :: Double -> Either String ProperFraction
constructProperFraction x =
    case prettyValidation $ ProperFraction x of
        Left err -> Left err
        Right y -> Right y

properToDouble :: ProperFraction -> Double
properToDouble (ProperFraction x) = x
