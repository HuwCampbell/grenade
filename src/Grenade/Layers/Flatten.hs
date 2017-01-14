{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleContexts      #-}
module Grenade.Layers.Flatten (
    FlattenLayer (..)
  ) where

import           Data.Singletons.TypeLits
import           GHC.TypeLits

import           Numeric.LinearAlgebra.Static
import           Numeric.LinearAlgebra.Data as LA ( flatten )

import           Grenade.Core.Shape
import           Grenade.Core.Network

-- | Flatten Layer
--
-- Flattens input down to D1 from either 2D or 3D data.
--
-- Can also be used to turn a 3D image with only one channel into a 2D image.
data FlattenLayer = FlattenLayer
  deriving Show

instance UpdateLayer FlattenLayer where
  type Gradient FlattenLayer = ()
  runUpdate _ _ _ = FlattenLayer
  createRandom = return FlattenLayer

instance (KnownNat a, KnownNat x, KnownNat y, a ~ (x * z)) => Layer FlattenLayer ('D2 x y) ('D1 a) where
  runForwards _ (S2D y)   = fromJust' . fromStorable . flatten . extract $ y
  runBackwards _ _ (S1D y) = ((), fromJust' . fromStorable . extract $ y)

instance (KnownNat a, KnownNat x, KnownNat y, KnownNat (x * z), KnownNat z, a ~ (x * y * z)) => Layer FlattenLayer ('D3 x y z) ('D1 a) where
  runForwards _ (S3D y)     = fromJust' . fromStorable . flatten . extract $ y
  runBackwards _ _ (S1D y) = ((), fromJust' . fromStorable . extract $ y)

instance (KnownNat y, KnownNat x, KnownNat z, z ~ 1) => Layer FlattenLayer ('D3 x y z) ('D2 x y) where
  runForwards _ (S3D y)    = S2D y
  runBackwards _ _ (S2D y) = ((), S3D y)

fromJust' :: Maybe x -> x
fromJust' (Just x) = x
fromJust' Nothing  = error $ "FlattenLayer error: data shape couldn't be converted."
