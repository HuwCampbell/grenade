{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeFamilies          #-}
{-|
Module      : Grenade.Core.Optimizer
Description : Defines the Optimizer classes
Copyright   : (c) Manuel Schneckenreither, 2020
License     : BSD2
Stability   : experimental
-}
module Grenade.Core.Optimizer
    ( Optimizer (..)
    , OptimizerAlgorithm (..)
    , defOptimizer
    ) where

import           Data.Default


-- | A kind used for instance declaration in the layer implementations.
--
--   Used only with DataKinds, as as Kind `OptimizerAlgorithm`, with Types 'SGD'.
data OptimizerAlgorithm = SGD

-- | The available optimizers to choose from. If an optimiser is not implemented for a layer SGD
--   with rate=0.01, momentum=0.9 and L2 regulariser=1e-4 will be used instead. This is default
--   and thus fallback optimizer.
--
--  Concreate instance for the optimizers.
data Optimizer (o :: OptimizerAlgorithm) where
  OptSGD
    :: { sgdLearningRate        :: Double -- ^ SGD Learning rate
       , sgdLearningMomentum    :: Double -- ^ SGD Momentum
       , sgdLearningRegulariser :: Double -- ^ SGD Regulasier (L2)
       }
    -> Optimizer 'SGD

instance Show (Optimizer o) where
  show (OptSGD r m l2) = "OptSGD" ++ show (r,m,l2)

-- | Default optimizer.
defOptimizer :: Optimizer 'SGD
defOptimizer = defSGD

-- | Default settings for the SGD optimizer.
instance Default (Optimizer 'SGD) where
  def = OptSGD 0.01 0.9 1e-4

-- | Default SGD optimizer.
defSGD :: Optimizer 'SGD
defSGD = def
