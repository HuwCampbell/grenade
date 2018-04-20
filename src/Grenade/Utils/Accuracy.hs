{-# LANGUAGE DeriveGeneric #-}

module Grenade.Utils.Accuracy
    ( Accuracy
    , accuracyM
    , showAccuracy
    ) where

import Data.Validity

import Data.Aeson (FromJSON, ToJSON)

import GHC.Generics

import Control.Monad.Catch

newtype Accuracy =
    Accuracy Double
    deriving (Show, Eq, Generic)

data AccuracyOutOfBounds =
    AccuracyOutOfBounds
    deriving (Show, Eq, Generic)

instance Exception AccuracyOutOfBounds where
    displayException = const "The accuracy is not in [0,1]."

accuracyM :: MonadThrow m => Double -> m Accuracy
accuracyM x =
    case prettyValidation $ Accuracy x of
        Right a -> pure a
        Left _ -> throwM AccuracyOutOfBounds

instance ToJSON Accuracy

instance FromJSON Accuracy

instance Validity Accuracy where
    validate (Accuracy x) =
        mconcat
            [ delve "The accuracy contains a valid double" x
            , declare "The accuracy is positive" $ x >= 0
            , declare "The accuracy is smaller than 1" $ x <= 1
            ]

showAccuracy :: String -> Accuracy -> String
showAccuracy name (Accuracy x) =
    "The " ++ name ++ " accuracy is " ++ show (nOfPrecent * x) ++ "%."

nOfPrecent :: Double
nOfPrecent = 100
