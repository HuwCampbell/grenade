{-|
Module      : Grenade.Core.LearningParameters
Description : Stochastic gradient descent learning parameters
Copyright   : (c) Huw Campbell, 2016-2017
License     : BSD2
Stability   : experimental
-}
module Grenade.Core.LearningParameters (
  -- | This module contains learning algorithm specific
  --   code. Currently, this module should be considered
  --   unstable, due to issue #26.

    LearningParameters (..)
  ) where

-- | Learning parameters for stochastic gradient descent.
data LearningParameters = LearningParameters {
    learningRate :: Double
  , learningMomentum :: Double
  , learningRegulariser :: Double
  } deriving (Eq, Show)
