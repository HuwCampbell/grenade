{-|
Module      : Grenade.Dynamic.Build
Description : Interface for building dynamic layers.
Copyright   : (c) Manuel Schneckenreither, 2020
License     : BSD2
Stability   : experimental
-}
module Grenade.Dynamic.Internal.Build
    ( DynamicBuildSetup (..)
    , BuildInfo (..)
    , BuildM
    -- builder tools for layer implementations
    , buildGetInfo
    , buildRequireLastLayerOut
    , buildGetLastLayerOut
    , buildSetLastLayer
    , buildAddSpec
    , DimensionsRequirement (..)
    ) where

import           Control.Monad.Reader
import           Control.Monad.State
import           Data.Default                  (Default (..))
import           Data.Maybe                    (fromMaybe)

import           Grenade.Dynamic.Specification


-- | Defines how to build the network.
newtype DynamicBuildSetup =
  DynamicBuildSetup
    { printResultingSpecification :: Bool -- ^ Print the result [Default: False]
    }

instance Default DynamicBuildSetup where
  def = DynamicBuildSetup False

-- | Build Info, only used internally.
data BuildInfo = BuildInfo
  { buildSpecs   :: ![SpecNet]
  , lastLayerOut :: !(Maybe Dimensions)
  }

type BuildM = StateT BuildInfo (Reader DynamicBuildSetup)

-- | Get the build info.
buildGetInfo :: BuildM BuildInfo
buildGetInfo = do
  info <- get
  case lastLayerOut info of
    Nothing -> error "A build specification must start with an input layer!"
    Just dim@(r, c, d) | r < 1 || c < 1 || d < 1 -> error $ "Every dimension has to be >=1, saw " ++ show dim
    Just _ -> return info

data DimensionsRequirement = Is1D | IsNot1D
  deriving (Eq, Ord)

-- | Assert a requirement of the last layer.
buildRequireLastLayerOut :: DimensionsRequirement -> BuildM Dimensions
buildRequireLastLayerOut expected = do
  info <- buildGetInfo
  case (expected, lastLayerOut info) of
    (_, Nothing) -> error "Programming error. You should have called buildGetInfo first!"
    (Is1D, Just dim) -> case dim of
      (_, 1, 1) -> return dim
      _         -> error $ "Expected 1D output, but saw " ++ show dim
    (IsNot1D, Just dim) -> case dim of
      (_, 1, 1) -> error $ "Expected 2D or 3D output, but saw " ++ show dim
      _         -> return dim

buildGetLastLayerOut :: BuildM Dimensions
buildGetLastLayerOut = fromMaybe err . lastLayerOut <$> buildGetInfo
  where err = error "Programming error, you should have called buildGetInfo!"

buildSetLastLayer :: Dimensions -> BuildM ()
buildSetLastLayer dim = modify $ \info -> info {lastLayerOut = Just dim}

buildAddSpec :: SpecNet -> BuildM ()
buildAddSpec spec = modify $ \info -> info { buildSpecs = spec : buildSpecs info }
