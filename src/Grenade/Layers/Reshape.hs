{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleContexts      #-}
module Grenade.Layers.Reshape (
    Reshape (..)
  ) where

import           Data.Serialize

import           Data.Singletons.TypeLits
import           GHC.TypeLits

import           Numeric.LinearAlgebra.Static
import           Numeric.LinearAlgebra.Data as LA ( flatten )

import           Grenade.Core

-- | Reshape Layer
--
-- Flattens input down to D1 from either 2D or 3D data.
--
-- Can also be used to turn a 3D image with only one channel into a 2D image.
data Reshape = Reshape
  deriving Show

instance UpdateLayer Reshape where
  type Gradient Reshape = ()
  runUpdate _ _ _ = Reshape
  createRandom = return Reshape

instance (KnownNat a, KnownNat x, KnownNat y, a ~ (x * y)) => Layer Reshape ('D2 x y) ('D1 a) where
  type Tape Reshape ('D2 x y) ('D1 a) = ()
  runForwards _ (S2D y)   =  ((), fromJust' . fromStorable . flatten . extract $ y)
  runBackwards _ _ (S1D y) = ((), fromJust' . fromStorable . extract $ y)

instance (KnownNat a, KnownNat x, KnownNat y, KnownNat (x * z), KnownNat z, a ~ (x * y * z)) => Layer Reshape ('D3 x y z) ('D1 a) where
  type Tape Reshape ('D3 x y z) ('D1 a) = ()
  runForwards _ (S3D y)     = ((), fromJust' . fromStorable . flatten . extract $ y)
  runBackwards _ _ (S1D y) = ((), fromJust' . fromStorable . extract $ y)

instance (KnownNat y, KnownNat x, KnownNat z, z ~ 1) => Layer Reshape ('D3 x y z) ('D2 x y) where
  type Tape Reshape ('D3 x y z) ('D2 x y) = ()
  runForwards _ (S3D y)    = ((), S2D y)
  runBackwards _ _ (S2D y) = ((), S3D y)

instance (KnownNat y, KnownNat x, KnownNat z, z ~ 1) => Layer Reshape ('D2 x y) ('D3 x y z) where
  type Tape Reshape ('D2 x y) ('D3 x y z) = ()
  runForwards _ (S2D y)    = ((), S3D y)
  runBackwards _ _ (S3D y) = ((), S2D y)

instance Serialize Reshape where
  put _ = return ()
  get = return Reshape


fromJust' :: Maybe x -> x
fromJust' (Just x) = x
fromJust' Nothing  = error $ "Reshape error: data shape couldn't be converted."
