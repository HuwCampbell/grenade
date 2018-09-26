{-|
Module      : Grenade.Core.LearningParameters
Description : Stochastic gradient descent learning parameters
Copyright   : (c) Huw Campbell, 2016-2017
License     : BSD2
Stability   : experimental
-}
module Grenade.Core.LearningParameters.Accelerate (
  -- | This module contains learning algorithm specific
  --   code. Currently, this module should be considered
  --   unstable, due to issue #26.

    LearningParameters
  ) where

import Data.Array.Accelerate

-- | Learning parameters for stochastic gradient descent.
type LearningParameters =
  ( Scalar Double -- learningRate
  , Scalar Double -- learningMomentum
  , Scalar Double -- learningRegulariser
  )
