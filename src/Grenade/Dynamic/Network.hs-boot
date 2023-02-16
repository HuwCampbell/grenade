{-# LANGUAGE CPP                   #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE InstanceSigs          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PolyKinds           #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE UndecidableInstances  #-}
{-|
Module      : Grenade.Layers.DynamicNetwork
Description : Dynamic grenade networks
Copyright   : (c) Manuel Schneckenreither, 2016-2020
License     : BSD2
Stability   : experimental

This module defines types and functions for dynamic generation of networks.
-}

module Grenade.Dynamic.Network
  ( SpecNetwork (..)
  , ToDynamicLayer (..)
  , SpecNet (..)
  , SpecFullyConnected (..)
  , SpecConvolution (..)
  , SpecDeconvolution (..)
  , SpecDropout (..)
  , SpecElu (..)
  , SpecLogit (..)
  , SpecRelu (..)
  , SpecReshape (..)
  , SpecSinusoid (..)
  , SpecSoftmax (..)
  , SpecTanh (..)
  , SpecTrivial (..)
  ) where

import           Control.DeepSeq
import           Control.Monad.Primitive           (PrimBase, PrimState)
import           Data.Serialize
import           Data.Singletons
import           Data.List.Singletons
import           Data.Typeable as T (Typeable)
import           System.Random.MWC
#if MIN_VERSION_base(4,9,0)
import           Data.Kind                         (Type)
#endif

import           Grenade.Core.Network
import           Grenade.Core.Layer
import           Grenade.Core.Shape
import           Grenade.Core.WeightInitialization
import           Grenade.Types


-- | Create a runtime dynamic specification of a network. Dynamic layers (and networks), for storing and restoring specific network structures (e.g. in saving the network structures to a DB and
-- restoring it from there) or simply generating them at runtime. This does not store the weights and biases! They have to be handled separately (see Serialize)!
class FromDynamicLayer x where
  fromDynamicLayer :: SomeSing Shape -> x -> SpecNet

-- | Class for generating layers from a specification.
class (Show spec) => ToDynamicLayer spec where
  toDynamicLayer :: (PrimBase m) => WeightInitMethod -> Gen (PrimState m) -> spec -> m SpecNetwork


----------------------------------------
-- Return value of toDynamicLayer

-- | Specification of a network or layer.
data SpecNetwork :: Type where
  SpecNetwork
    :: (SingI shapes, SingI (Head shapes), SingI (Last shapes), Show (Network layers shapes), FromDynamicLayer (Network layers shapes), NFData (Network layers shapes)
       , Layer (Network layers shapes) (Head shapes) (Last shapes), RandomLayer (Network layers shapes), Serialize (Network layers shapes), GNum (Network layers shapes)
       , NFData (Tapes layers shapes), GNum (Gradients layers))
    => !(Network layers shapes)
    -> SpecNetwork
  SpecLayer :: (FromDynamicLayer x, RandomLayer x, Serialize x, NFData (Tape x i o), GNum (Gradient x), GNum x, NFData x, Show x, Layer x i o) => !x -> !(Sing i) -> !(Sing o) -> SpecNetwork


----------------------------------------
-- Specification of a network and its layers

-- | Data structure for holding specifications for networks. Networks can be built dynamically with @toDynamicLayer@. Further see the functions @|=>@, @specNil1D@, @specNil2D@, @specNil3D@, and
-- possibly any other layer implementation of @ToDynamicLayer@ for building specifications.
data SpecNet
  = SpecNNil1D !Integer                   -- ^ 1D network output
  | SpecNNil2D !Integer !Integer          -- ^ 2D network output
  | SpecNNil3D !Integer !Integer !Integer -- ^ 3D network output
  | SpecNCons !SpecNet !SpecNet           -- ^ x :~> xs, where x can also be a network
  | forall spec . (ToDynamicLayer spec, Typeable spec, Ord spec, Eq spec, Show spec, Serialize spec, NFData spec) => SpecNetLayer !spec -- ^ Specification of a layer

type Dimensions = (Integer, Integer, Integer)

-- Data structures instances for Layers (needs to be defined here)

data SpecFullyConnected = SpecFullyConnected !Integer !Integer

-- data SpecConcat = SpecConcat SpecNet SpecNet

data SpecConvolution =
  SpecConvolution !Dimensions !Integer !Integer !Integer !Integer !Integer !Integer

data SpecDeconvolution =
  SpecDeconvolution !Dimensions !Integer !Integer !Integer !Integer !Integer !Integer

data SpecDropout = SpecDropout !Integer !RealNum !(Maybe Int)

newtype SpecElu = SpecElu Dimensions

newtype SpecLogit = SpecLogit Dimensions

newtype SpecRelu = SpecRelu Dimensions

data SpecReshape = SpecReshape !Dimensions !Dimensions

newtype SpecSinusoid = SpecSinusoid Dimensions

newtype SpecSoftmax = SpecSoftmax Integer

newtype SpecTanh = SpecTanh Dimensions

newtype SpecTrivial = SpecTrivial Dimensions
