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

module Grenade.Core.DynamicNetwork
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
  , SpecSinusoid (..)
  , SpecSoftmax (..)
  , SpecTanh (..)
  , SpecTrivial (..)
  ) where

import           Control.DeepSeq
import           Control.Monad.Primitive           (PrimBase, PrimState)
import           Data.Constraint                   (Dict (..))
import           Data.Reflection (reifyNat)
import           Data.Typeable as T (typeOf, Typeable, cast)
import           Data.Serialize
import           Data.Singletons
import           Data.Singletons.TypeLits (SNat (..))
import           Data.Singletons.Prelude
import           GHC.TypeLits
import           GHC.Generics
import           System.Random.MWC
import           Unsafe.Coerce                     (unsafeCoerce)
#if MIN_VERSION_base(4,9,0)
import           Data.Kind                         (Type)
#endif

import           Grenade.Core.Network
import           Grenade.Core.Layer
import           Grenade.Core.Shape
import           Grenade.Core.WeightInitialization

import Debug.Trace

-- | Create a runtime dynamic specification of a network. Dynamic layers (and networks), for storing and restoring specific network structures (e.g. in saving the network structures to a DB and
-- restoring it from there) or simply generating them at runtime. This does not store the weights and biases! They have to be handled separately (see Serialize)!
class FromDynamicLayer x where
  fromDynamicLayer :: SomeSing Shape -> x -> SpecNet

-- | Class for generating layers from a specification.
class (Show spec) => ToDynamicLayer spec where
  toDynamicLayer :: (PrimBase m) => WeightInitMethod -> Gen (PrimState m) -> spec -> m SpecNetwork


----------------------------------------
-- Return value of toDynamicLayer

data SpecNetwork :: Type where
  SpecNetwork
    :: (SingI shapes, SingI (Head shapes), SingI (Last shapes), Show (Network layers shapes), FromDynamicLayer (Network layers shapes), NFData (Network layers shapes)
       , Layer (Network layers shapes) (Head shapes) (Last shapes), RandomLayer (Network layers shapes)
       )
    => !(Network layers shapes)
    -> SpecNetwork
  SpecLayer :: (FromDynamicLayer x, RandomLayer x, NFData x, Show x, Layer x i o) => !x -> !(Sing i) -> !(Sing o) -> SpecNetwork


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


-- Data structures instances for Layers (needs to be defined here)

data SpecFullyConnected = SpecFullyConnected Integer Integer

-- data SpecConcat = SpecConcat SpecNet SpecNet

data SpecConvolution =
  SpecConvolution (Integer, Integer, Integer) Integer Integer Integer Integer Integer Integer

data SpecDeconvolution =
  SpecDeconvolution (Integer, Integer, Integer) Integer Integer Integer Integer Integer Integer

data SpecDropout = SpecDropout Integer Double (Maybe Int)

newtype SpecElu = SpecElu (Integer, Integer, Integer)

newtype SpecLogit = SpecLogit (Integer, Integer, Integer)

newtype SpecRelu = SpecRelu (Integer, Integer, Integer)

newtype SpecSinusoid = SpecSinusoid (Integer, Integer, Integer)

newtype SpecSoftmax = SpecSoftmax Integer

newtype SpecTanh = SpecTanh (Integer, Integer, Integer)

newtype SpecTrivial = SpecTrivial (Integer, Integer, Integer)


