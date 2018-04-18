{-|
Module      : Grenade.Core.LearningParameters
Description : Stochastic gradient descent learning parameters
Copyright   : (c) Huw Campbell, 2016-2017
License     : BSD2
Stability   : experimental
-}

{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}

module Grenade.Core.LearningParameters (
  -- | This module contains learning algorithm specific
  --   code. Currently, this module should be considered
  --   unstable, due to issue #26.

    LearningParameters (..)
  ) where

import GHC.Generics

import Data.Aeson (ToJSON, FromJSON)

import Data.Validity

-- | Learning parameters for stochastic gradient descent.
data LearningParameters = LearningParameters {
    learningRate :: Double
  , learningMomentum :: Double
  , learningRegulariser :: Double
  } deriving (Eq, Show, Generic)

instance ToJSON LearningParameters

instance FromJSON LearningParameters

instance Validity LearningParameters where
    validate LearningParameters {..} =
        mconcat
            [ learningRate > 0 <?@> "The learning rate must be strictly positive"
            , learningMomentum >= 0 <?@> "The momentum parameter must be positive"
            , learningRegulariser >= 0 <?@> "The regulariser must be positive"
            ]
