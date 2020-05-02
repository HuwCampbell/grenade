{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE CPP #-}
{-# LANGUAGE DeriveGeneric  #-}
module Grenade.Dynamic.Specification
  ( ToDynamicLayer (..)
  , SpecFullyConnected(..)
  , SpecConvolution(..)
  , SpecDeconvolution(..)
  , SpecDropout(..)
  , SpecElu(..)
  , SpecLogit(..)
  , SpecRelu(..)
  , SpecReshape(..)
  , SpecSinusoid(..)
  , SpecSoftmax(..)
  , SpecTanh(..)
  , SpecTrivial(..)
  ) where

import           Control.DeepSeq
import           Data.Serialize
import           Data.Typeable as T
import           Data.Singletons
import           Data.Singletons.Prelude.List
#if MIN_VERSION_base(4,9,0)
import           Data.Kind                         (Type)
#endif


import           Grenade.Types
import           Grenade.Core
import           Control.Monad.Primitive           (PrimBase, PrimState)
import           System.Random.MWC (Gen)

class FromDynamicLayer x where
  fromDynamicLayer :: SomeSing Shape -> SomeSing Shape -> x -> SpecNet

class (Show spec) => ToDynamicLayer spec where
  toDynamicLayer :: (PrimBase m) => WeightInitMethod -> Gen (PrimState m) -> spec -> m SpecNetwork


----------------------------------------
-- Return value of toDynamicLayer

-- | Specification of a network or layer.
data SpecNetwork :: Type where
  SpecNetwork
    :: ( FromDynamicLayer (Network layers shapes)
       , FoldableGradient (Gradient (Network layers shapes))
       , GNum (Gradients layers)
       , GNum (Network layers shapes)
       , Layer (Network layers shapes) (Head shapes) (Last shapes)
       , NFData (Gradients layers)
       , NFData (Network layers shapes)
       , NFData (Tapes layers shapes)
       , RandomLayer (Network layers shapes)
       , Serialize (Network layers shapes)
       , Serialize (Gradients layers)
       , Show (Network layers shapes)
       , SingI (Head shapes)
       , SingI (Last shapes)
       , SingI shapes
       , Typeable (Head shapes)
       , Typeable layers
       , Typeable shapes
       )
    => !(Network layers shapes)
    -> SpecNetwork
  SpecLayer
    :: ( FromDynamicLayer x
       , FoldableGradient (Gradient x)
       , GNum (Gradient x)
       , GNum x
       , Layer x i o
       , NFData (Gradient x)
       , NFData (Tape x i o)
       , NFData x
       , RandomLayer x
       , Serialize x
       , Serialize (Gradient x)
       , Show x
       , Typeable i
       , Typeable x
       )
    => !x
    -> !(Sing i)
    -> !(Sing o)
    -> SpecNetwork

data SpecNet
  = SpecNNil1D !Integer                   -- ^ 1D network output
  | SpecNNil2D !Integer !Integer          -- ^ 2D network output
  | SpecNNil3D !Integer !Integer !Integer -- ^ 3D network output
  | SpecNCons !SpecNet !SpecNet           -- ^ x :~> xs, where x can also be a network
  | forall spec . (ToDynamicLayer spec, Typeable spec, Ord spec, Eq spec, Show spec, Serialize spec, NFData spec) => SpecNetLayer !spec -- ^ Specification of a layer


-- Data structures stances for Layers (needs to be defined here)

data SpecFullyConnected = SpecFullyConnected !Integer !Integer

-- data SpecConcat = SpecConcat SpecNet SpecNet

data SpecConvolution =
  SpecConvolution !(Integer, Integer, Integer) !Integer !Integer !Integer !Integer !Integer !Integer

data SpecDeconvolution =
  SpecDeconvolution !(Integer, Integer, Integer) !Integer !Integer !Integer !Integer !Integer !Integer

data SpecDropout = SpecDropout !Integer !RealNum !(Maybe Int)

newtype SpecElu = SpecElu (Integer, Integer, Integer)

newtype SpecLogit = SpecLogit (Integer, Integer, Integer)

newtype SpecRelu = SpecRelu (Integer, Integer, Integer)

data SpecReshape = SpecReshape !(Integer, Integer, Integer) !(Integer, Integer, Integer)

newtype SpecSinusoid = SpecSinusoid (Integer, Integer, Integer)

newtype SpecSoftmax = SpecSoftmax Integer

newtype SpecTanh = SpecTanh (Integer, Integer, Integer)

newtype SpecTrivial = SpecTrivial (Integer, Integer, Integer)
