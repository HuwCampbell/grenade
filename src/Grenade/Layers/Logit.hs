{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
module Grenade.Layers.Logit (
    Logit (..)
  ) where


import           Data.Serialize

import           Data.Singletons.TypeLits
import           Grenade.Core.Network
import           Grenade.Core.Shape

-- | A Logit layer.
--   A layer which can act between any shape of the same dimension, perfoming an sigmoid function.
--   This layer should be used as the output layer of a network for logistic regression (classification)
--   problems.
data Logit = Logit
  deriving Show

instance UpdateLayer Logit where
  type Gradient Logit = ()
  runUpdate _ _ _ = Logit
  createRandom = return Logit

instance (KnownNat i) => Layer Logit ('D1 i) ('D1 i) where
  type Tape Logit ('D1 i) ('D1 i) = S ('D1 i)
  runForwards _ (S1D y) = (S1D y, S1D (logistic y))
  runBackwards _ (S1D y) (S1D dEdy) = ((), S1D (logistic' y * dEdy))

instance (KnownNat i, KnownNat j) => Layer Logit ('D2 i j) ('D2 i j) where
  type Tape Logit ('D2 i j) ('D2 i j) = S ('D2 i j)
  runForwards _ (S2D y) = (S2D y, S2D (logistic y))
  runBackwards _ (S2D y) (S2D dEdy) = ((), S2D (logistic' y * dEdy))

instance (KnownNat i, KnownNat j, KnownNat k) => Layer Logit ('D3 i j k) ('D3 i j k) where
  type Tape Logit ('D3 i j k) ('D3 i j k) = S ('D3 i j k)
  runForwards _ (S3D y) = (S3D y, S3D (logistic y))
  runBackwards _ (S3D y) (S3D dEdy) = ((), S3D (logistic' y * dEdy))

instance Serialize Logit where
  put _ = return ()
  get = return Logit

logistic :: Floating a => a -> a
logistic x = 1 / (1 + exp (-x))

logistic' :: Floating a => a -> a
logistic' x = logix * (1 - logix)
  where
    logix = logistic x
