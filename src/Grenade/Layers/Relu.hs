{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE StandaloneDeriving    #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances     #-}

module Grenade.Layers.Relu (
    Relu (..)
  ) where

import           GHC.TypeLits
import           Grenade.Core.Vector
import           Grenade.Core.Network
import           Grenade.Core.Shape

import qualified Numeric.LinearAlgebra.Static as LAS

-- | A rectifying linear unit.
--   A layer which can act between any shape of the same dimension, acting as a
--   diode on every neuron individually.
data Relu = Relu
  deriving Show

instance (Monad m, KnownNat i) => Layer m Relu ('D1 i) ('D1 i) where
  runForwards _ (S1D' y) = return $ S1D' (relu y)
    where
      relu = LAS.dvmap (\a -> if a <= 0 then 0 else a)
  runBackards _ _ (S1D' y) (S1D' dEdy) = return (Relu, S1D' (relu' y * dEdy))
    where
      relu' = LAS.dvmap (\a -> if a <= 0 then 0 else 1)

instance (Monad m, KnownNat i, KnownNat j) => Layer m Relu ('D2 i j) ('D2 i j) where
  runForwards _ (S2D' y) = return $ S2D' (relu y)
    where
      relu = LAS.dmmap (\a -> if a <= 0 then 0 else a)
  runBackards _ _ (S2D' y) (S2D' dEdy) = return (Relu, S2D' (relu' y * dEdy))
    where
      relu' = LAS.dmmap (\a -> if a <= 0 then 0 else 1)

instance (Monad m, KnownNat i, KnownNat j, KnownNat k) => Layer m Relu ('D3 i j k) ('D3 i j k) where
  runForwards _ (S3D' y) = return $ S3D' (fmap relu y)
    where
      relu = LAS.dmmap (\a -> if a <= 0 then 0 else a)
  runBackards _ _ (S3D' y) (S3D' dEdy) = return (Relu, S3D' (vectorZip (\y' dEdy' -> relu' y' * dEdy') y dEdy))
    where
      relu' = LAS.dmmap (\a -> if a <= 0 then 0 else 1)
