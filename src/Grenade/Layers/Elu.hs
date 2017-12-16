{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-|
Module      : Grenade.Layers.Logit
Description : Exponential linear unit layer
Copyright   : (c) Huw Campbell, 2016-2017
License     : BSD2
Stability   : experimental
-}
module Grenade.Layers.Elu (
    Elu (..)
  ) where

import           Data.Serialize

import           GHC.TypeLits
import           Grenade.Core

import qualified Numeric.LinearAlgebra.Static as LAS

-- | An exponential linear unit.
--   A layer which can act between any shape of the same dimension, acting as a
--   diode on every neuron individually.
data Elu = Elu
  deriving Show

instance UpdateLayer Elu where
  type Gradient Elu = ()
  runUpdate _ _ _ = Elu
  createRandom = return Elu

instance Serialize Elu where
  put _ = return ()
  get = return Elu

instance ( KnownNat i) => Layer Elu ('D1 i) ('D1 i) where
  type Tape Elu ('D1 i) ('D1 i) = LAS.R i

  runForwards _ (S1D y) = (y, S1D (elu y))
    where
      elu = LAS.dvmap (\a -> if a <= 0 then exp a - 1 else a)
  runBackwards _ y (S1D dEdy) = ((), S1D (elu' y * dEdy))
    where
      elu' = LAS.dvmap (\a -> if a <= 0 then exp a else 1)

instance (KnownNat i, KnownNat j) => Layer Elu ('D2 i j) ('D2 i j) where
  type Tape Elu ('D2 i j) ('D2 i j) = S ('D2 i j)

  runForwards _ (S2D y) = (S2D y, S2D (elu y))
    where
      elu = LAS.dmmap (\a -> if a <= 0 then exp a - 1 else a)
  runBackwards _ (S2D y) (S2D dEdy) = ((), S2D (elu' y * dEdy))
    where
      elu' = LAS.dmmap (\a -> if a <= 0 then exp a else 1)

instance (KnownNat i, KnownNat j, KnownNat k) => Layer Elu ('D3 i j k) ('D3 i j k) where

  type Tape Elu ('D3 i j k) ('D3 i j k) = S ('D3 i j k)

  runForwards _ (S3D y) = (S3D y, S3D (elu y))
    where
      elu = LAS.dmmap (\a -> if a <= 0 then exp a - 1 else a)
  runBackwards _ (S3D y) (S3D dEdy) = ((), S3D (elu' y * dEdy))
    where
      elu' = LAS.dmmap (\a -> if a <= 0 then exp a else 1)
