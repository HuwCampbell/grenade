{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances     #-}

module Grenade.Layers.Tanh (
    Tanh (..)
  ) where

import           GHC.TypeLits
import           Grenade.Core.Vector
import           Grenade.Core.Network
import           Grenade.Core.Shape

-- | A Tanh layer.
--   A layer which can act between any shape of the same dimension, perfoming a tanh function.
data Tanh = Tanh
  deriving Show

instance UpdateLayer Tanh where
  type Gradient Tanh = ()
  runUpdate _ _ _ = Tanh
  createRandom = return Tanh

instance KnownNat i => Layer Tanh ('D1 i) ('D1 i) where
  runForwards _ (S1D' y) = S1D' (tanh y)
  runBackards _ (S1D' y) (S1D' dEdy) = ((), S1D' (tanh' y * dEdy))

instance (KnownNat i, KnownNat j) => Layer Tanh ('D2 i j) ('D2 i j) where
  runForwards _ (S2D' y) =  S2D' (tanh y)
  runBackards _ (S2D' y) (S2D' dEdy) = ((), S2D' (tanh' y * dEdy))

instance (KnownNat i, KnownNat j, KnownNat k) => Layer Tanh ('D3 i j k) ('D3 i j k) where
  runForwards _ (S3D' y) = S3D' (fmap tanh y)
  runBackards _ (S3D' y) (S3D' dEdy) = ((), S3D' (vectorZip (\y' dEdy' -> tanh' y' * dEdy') y dEdy))

tanh' :: (Floating a) => a -> a
tanh' t = 1 - s ^ (2 :: Int)  where s = tanh t
