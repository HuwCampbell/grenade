{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-|
Module      : Grenade.Layers.Relu
Description : Rectifying linear unit layer
Copyright   : (c) Huw Campbell, 2016-2017
License     : BSD2
Stability   : experimental
-}
module Grenade.Layers.Relu (
    Relu (..)
  ) where

import           Data.Serialize

import           GHC.TypeLits
import           Grenade.Core

import qualified Numeric.LinearAlgebra.Static as LAS

-- | A rectifying linear unit.
--   A layer which can act between any shape of the same dimension, acting as a
--   diode on every neuron individually.
data Relu = Relu
  deriving Show

instance UpdateLayer Relu where
  type Gradient Relu = ()
  runUpdate _ _ _ = Relu
  createRandom = return Relu

instance Serialize Relu where
  put _ = return ()
  get = return Relu

instance ( KnownNat i) => Layer Relu ('D1 i) ('D1 i) where
  type Tape Relu ('D1 i) ('D1 i) = S ('D1 i)

  runForwards _ (S1D y) = (S1D y, S1D (relu y))
    where
      relu = LAS.dvmap (\a -> if a <= 0 then 0 else a)
  runBackwards _ (S1D y) (S1D dEdy) = ((), S1D (relu' y * dEdy))
    where
      relu' = LAS.dvmap (\a -> if a <= 0 then 0 else 1)

instance (KnownNat i, KnownNat j) => Layer Relu ('D2 i j) ('D2 i j) where
  type Tape Relu ('D2 i j) ('D2 i j) = S ('D2 i j)

  runForwards _ (S2D y) = (S2D y, S2D (relu y))
    where
      relu = LAS.dmmap (\a -> if a <= 0 then 0 else a)
  runBackwards _ (S2D y) (S2D dEdy) = ((), S2D (relu' y * dEdy))
    where
      relu' = LAS.dmmap (\a -> if a <= 0 then 0 else 1)

instance (KnownNat i, KnownNat j, KnownNat k) => Layer Relu ('D3 i j k) ('D3 i j k) where

  type Tape Relu ('D3 i j k) ('D3 i j k) = S ('D3 i j k)

  runForwards _ (S3D y) = (S3D y, S3D (relu y))
    where
      relu = LAS.dmmap (\a -> if a <= 0 then 0 else a)
  runBackwards _ (S3D y) (S3D dEdy) = ((), S3D (relu' y * dEdy))
    where
      relu' = LAS.dmmap (\a -> if a <= 0 then 0 else 1)
