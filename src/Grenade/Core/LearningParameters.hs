module Grenade.Core.LearningParameters (
  -- | This module contains learning algorithm specific
  --   code. Currently, this module should be consifered
  --   unstable, due to issue #26.

    LearningParameters (..)
  ) where

-- | Learning parameters for stochastic gradient descent.
data LearningParameters = LearningParameters {
    learningRate :: Double
  , learningMomentum :: Double
  , learningRegulariser :: Double
  } deriving (Eq, Show)
