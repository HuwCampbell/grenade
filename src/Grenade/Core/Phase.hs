{-# LANGUAGE FlexibleContexts      #-}
module Grenade.Core.Phase (
    Phase (..)
  , isTrainingPhase
  ) where

import           Control.Monad.State

data Phase = TrainingPhase
           | TestingPhase
           deriving (Eq, Ord, Show)

isTrainingPhase :: MonadState Phase m => m Bool
isTrainingPhase = (== TrainingPhase) <$> get
