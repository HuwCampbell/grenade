{-# LANGUAGE CPP                   #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PolyKinds             #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE UndecidableInstances  #-}
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
    , defSGD
    , defAdam
    ) where

import           Data.Default
import           Data.Kind       (Type)
import           Data.Serialize
import           Data.Singletons

import           Grenade.Types

-- | A kind used for instance declaration in the layer implementations.
--
--   Used only with DataKinds, as as Kind `OptimizerAlgorithm`, with Types 'SGD'.
data OptimizerAlgorithm = SGD | Adam

-- | The available optimizers to choose from. If an optimiser is not implemented for a layer SGD
--   with the default settings (see @Default 'SGD@ instance) will be used instead. This is default
--   and thus fallback optimizer.
--
--  Concreate instance for the optimizers.
data Optimizer (o :: OptimizerAlgorithm) where
  OptSGD
    :: { sgdLearningRate        :: !F -- ^ SGD Learning rate [Default: 0.01]
       , sgdLearningMomentum    :: !F -- ^ SGD Momentum [Default 0.9]
       , sgdLearningRegulariser :: !F -- ^ SGD Regulasier (L2) [Default: 0.0001]
       }
    -> Optimizer 'SGD
  OptAdam
    :: { adamAlpha   :: !F -- ^ Alpha [Default: 0.001]
       , adamBeta1   :: !F -- ^ Beta 1 [Default: 0.9]
       , adamBeta2   :: !F -- ^ Beta 2 [Default: 0.999]
       , adamEpsilon :: !F -- ^ Epsilon [Default: 1e-7]
       }
    -> Optimizer 'Adam

-- | Default optimizer.
defOptimizer :: Optimizer 'SGD
defOptimizer = defSGD

-- | Default settings for the SGD optimizer.
instance Default (Optimizer 'SGD) where
  def = OptSGD 0.01 0.9 1e-4

-- | Default settings for the SGD optimizer.
instance Default (Optimizer 'Adam) where
  def = OptAdam 0.001 0.9 0.999 1e-7

-- | Default SGD optimizer.
defSGD :: Optimizer 'SGD
defSGD = def

defAdam :: Optimizer 'Adam
defAdam = def


-- instances

instance Show (Optimizer o) where
  show (OptSGD r m l2) = "SGD" ++ show (r, m, l2)
  show (OptAdam alpha beta1 beta2 epsilon) = "Adam" ++ show (alpha, beta1, beta2, epsilon)

type instance Sing = Opt

data Opt (opt :: OptimizerAlgorithm) where
  SSGD :: Opt 'SGD
  SAdam :: Opt 'Adam

instance SingI opt => Serialize (Optimizer opt) where
  put (OptSGD rate m reg) = put rate >> put m >> put reg
  put (OptAdam a b1 b2 e) = put a >> put b1 >> put b2 >> put e
  get =
    case sing :: Opt opt of
      SSGD  -> OptSGD <$> get <*> get <*> get
      SAdam -> OptAdam <$> get <*> get <*> get <*> get

