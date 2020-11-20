{-|
Module      : Grenade.Dynamic.Build
Description : Interface for building dynamic layers.
Copyright   : (c) Manuel Schneckenreither, 2020
License     : BSD2
Stability   : experimental
-}
module Grenade.Dynamic.Build
    ( DynamicBuildSetup (..)
    , BuildM
    , buildModel
    , buildModelWith
    -- model builer functions
    , inputLayer
    , inputLayer1D
    , inputLayer2D
    , inputLayer3D
    , networkLayer
    ) where

import           Control.Monad.Reader
import           Control.Monad.State
import           Data.Default                     (Default (..))
import           Data.List                        (foldl')
import           Data.Maybe                       (fromMaybe)

import           Grenade.Core.NetworkInitSettings
import           Grenade.Dynamic.Internal.Build
import           Grenade.Dynamic.Network
import           Grenade.Dynamic.Specification


-- | Input layer, for unused dimensions specify 1, e.g. (1,1,1) is 1D, (1,2,1) is 2D, (3,4,1) is 2D, and (2,1,2) is 3D.
inputLayer :: (Integer, Integer, Integer) -> BuildM ()
inputLayer = buildSetLastLayer

inputLayer1D :: Integer -> BuildM ()
inputLayer1D r = inputLayer (r, 1, 1)
inputLayer2D :: (Integer, Integer) -> BuildM ()
inputLayer2D (r, c) = inputLayer (r, c, 1)
inputLayer3D :: Dimensions -> BuildM ()
inputLayer3D = inputLayer

-- | Add a network as sublayer.
networkLayer :: BuildM () -> BuildM ()
networkLayer build = do
  setup <- ask
  let (info, spec) = buildSpec setup build
  buildAddSpec $ SpecNetLayer spec
  buildSetLastLayer $ fromMaybe err (lastLayerOut info)
  where
    err = error "Unexpected empty sub-network dimensions"


-- | Build a model specified in the BuildM monad using UniformInit as weight initialization method.
buildModel :: BuildM () -> IO SpecConcreteNetwork
buildModel = buildModelWith def def


-- | Build a model specified in the BuildM monad using the given weight initialization method.
buildModelWith :: NetworkInitSettings -> DynamicBuildSetup -> BuildM () -> IO SpecConcreteNetwork
buildModelWith wInit setup builder = do
  let spec = snd $ buildSpec setup builder
  when (printResultingSpecification setup) $ do
    putStrLn "Building following model specification: "
    print spec
  networkFromSpecificationWith wInit spec


-- | Build a specification.
buildSpec :: DynamicBuildSetup -> BuildM () -> (BuildInfo, SpecNet)
buildSpec setup builder =
  let buildInfo = runReader (execStateT builder emptyBuildInfo) setup
      mkSpecNet = foldl' (flip SpecNCons)
      spec = case lastLayerOut buildInfo of
              Nothing        -> error "Empty input"
              Just (r, 1, 1) -> mkSpecNet (SpecNNil1D r) (buildSpecs buildInfo)
              Just (r, c, 1) -> mkSpecNet (SpecNNil2D r c) (buildSpecs buildInfo)
              Just (r, c, d) -> mkSpecNet (SpecNNil3D r c d) (buildSpecs buildInfo)
  in (buildInfo, spec)
  where
    emptyBuildInfo = BuildInfo [] Nothing
