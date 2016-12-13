{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE StandaloneDeriving    #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE FlexibleInstances     #-}

module Grenade.Layers.Flatten (
    FlattenLayer (..)
  ) where

import           Data.Singletons.TypeLits
import           GHC.TypeLits

import           Numeric.LinearAlgebra.Static
import           Numeric.LinearAlgebra.Data as  LA (flatten, toList)

import           Grenade.Core.Shape
import           Grenade.Core.Network

data FlattenLayer = FlattenLayer
  deriving Show

instance UpdateLayer FlattenLayer where
  type Gradient FlattenLayer = ()
  runUpdate _ _ _ = FlattenLayer
  createRandom = return FlattenLayer


instance (KnownNat a, KnownNat x, KnownNat y, a ~ (x * z)) => Layer FlattenLayer ('D2 x y) ('D1 a) where
  runForwards _ (S2D' y)   = S1D' . fromList . toList . flatten . extract $ y
  runBackwards _ _ (S1D' y) = ((), S2D' . fromList . toList . unwrap $ y)

instance (KnownNat a, KnownNat x, KnownNat y, KnownNat (x * z), KnownNat z, a ~ (x * y * z)) => Layer FlattenLayer ('D3 x y z) ('D1 a) where
  runForwards _ (S3D' y)     = S1D' . fromList . toList . flatten . extract $ y
  runBackwards _ _ (S1D' y) = ((), S3D' . fromList . toList . unwrap $ y)
