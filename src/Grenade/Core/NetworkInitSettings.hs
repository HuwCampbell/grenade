{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric  #-}
{-|
Module      : Grenade.Core.NetworkInitSettings
Description : Defines the initialization parameters for the network initialization
Copyright   : (c) Manuel Schneckenreither, 2020
License     : BSD2
Stability   : experimental

-}
module Grenade.Core.NetworkInitSettings
    ( NetworkInitSettings(..)
    , CPUBackend (..)
    ) where

import           Control.DeepSeq
import           Data.Default
import           Data.Serialize
import           GHC.Generics

import           Grenade.Core.WeightInitialization

-- | Defines the backend that is used for CPU vector/matrix computations.
data CPUBackend
  = HMatrix -- ^ Using HMatrix package, which is the default implementation and uses BLAS/LAPACK, but is slow for larger matrices.
  | CBLAS   -- ^ Directly using CBLAS/LAPACK.
  deriving (Show, Eq, Ord, Enum, Bounded, NFData, Serialize, Generic)


-- | Configures the initialization phase of the ANN.
data NetworkInitSettings = NetworkInitSettings
  { weightInitMethod :: WeightInitMethod -- ^ Initialization method for random numbers.
  , cpuBackend       :: CPUBackend       -- ^ Backend for CPU vector/matrix computations.
  } deriving (Show, Eq, Ord, NFData, Serialize, Generic)


instance Default NetworkInitSettings where
  def = NetworkInitSettings UniformInit HMatrix
