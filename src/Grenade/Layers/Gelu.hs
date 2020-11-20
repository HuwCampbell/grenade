{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE DeriveAnyClass        #-}
{-# LANGUAGE DeriveGeneric         #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TypeOperators         #-}
{-|
Module      : Grenade.Layers.Gelu
Description : Gaussian Error Linear Unit (GELU)
Copyright   : (c) Manuel Schneckenreither, 2020
License     : BSD2
Stability   : experimental

This module implements the Gaussian Error Linear Unit (GELU) activiation function. See

Hendrycks, Dan, and Kevin Gimpel. "Gaussian error linear units (gelus)." arXiv preprint arXiv:1606.08415 (2016).

Available at: https://arxiv.org/pdf/1606.08415.pdf

As in the paper we simply let μ = 0 and σ = 1. Futher, we use the simplified and thus fast representation: x * sigmoid (1.702 * x)

-}
module Grenade.Layers.Gelu (
    Gelu (..)
  , SpecGelu (..)
  , specGelu1D
  , specGelu2D
  , specGelu3D
  , gelu
  ) where

import           Control.DeepSeq                (NFData (..))
import           Data.Constraint                (Dict (..))
import           Data.Reflection                (reifyNat)
import           Data.Serialize
import           Data.Singletons
import           GHC.Generics                   (Generic)
import           GHC.TypeLits
import qualified Numeric.LinearAlgebra.Static   as LAS
import           Unsafe.Coerce                  (unsafeCoerce)

import           Grenade.Core
import           Grenade.Dynamic
import           Grenade.Dynamic.Internal.Build


-- | A Gaussion Error Linear Unit.
--   A layer which can act between any shape of the same dimension, acting as a
--   diode on every neuron individually.
--
--   Hendrycks, Dan, and Kevin Gimpel. "Gaussian error linear units (gelus)." arXiv preprint arXiv:1606.08415 (2016).
data Gelu = Gelu
  deriving (Generic, NFData, Show)

instance UpdateLayer Gelu where
  type Gradient Gelu = ()
  runUpdate _ _ _ = Gelu

instance RandomLayer Gelu where
  createRandomWith _ _ = return Gelu

instance Serialize Gelu where
  put _ = return ()
  get = return Gelu

geluForwardFast :: Floating x => x -> x
geluForwardFast x = x / (e ** (-1.702 * x) + 1) -- = x * sigmoid (1.702 * x)
  where
    e = 2.71828

geluBackwardFast :: Floating x => x -> x
geluBackwardFast x = (e ** (1.702 * x) * (1 + e ** (1.702 * x) + 1.702 * x)) / (1 + e ** (1.702 * x)) ** 2
  where
    e = 2.71828

instance (KnownNat i) => Layer Gelu ('D1 i) ('D1 i) where
  type Tape Gelu ('D1 i) ('D1 i) = S ('D1 i)

  runForwards _ (S1D y) = (S1D y, S1D (gelu y))
    where
      gelu = LAS.dvmap geluForwardFast
  runBackwards _ (S1D y) (S1D dEdy) = ((), S1D (gelu' y * dEdy))
    where
      gelu' = LAS.dvmap geluBackwardFast

instance (KnownNat i, KnownNat j) => Layer Gelu ('D2 i j) ('D2 i j) where
  type Tape Gelu ('D2 i j) ('D2 i j) = S ('D2 i j)

  runForwards _ (S2D y) = (S2D y, S2D (gelu y))
    where
      gelu = LAS.dmmap geluForwardFast
  runBackwards _ (S2D y) (S2D dEdy) = ((), S2D (gelu' y * dEdy))
    where
      gelu' = LAS.dmmap geluBackwardFast

instance (KnownNat i, KnownNat j, KnownNat k) => Layer Gelu ('D3 i j k) ('D3 i j k) where

  type Tape Gelu ('D3 i j k) ('D3 i j k) = S ('D3 i j k)

  runForwards _ (S3D y) = (S3D y, S3D (gelu y))
    where
      gelu = LAS.dmmap geluForwardFast
  runBackwards _ (S3D y) (S3D dEdy) = ((), S3D (gelu' y * dEdy))
    where
      gelu' = LAS.dmmap geluBackwardFast


-------------------- DynamicNetwork instance --------------------

instance FromDynamicLayer Gelu where
  fromDynamicLayer inp _ _ = SpecNetLayer $ SpecGelu (tripleFromSomeShape inp)

instance ToDynamicLayer SpecGelu where
  toDynamicLayer _ _ (SpecGelu (rows, cols, depth)) =
     reifyNat rows $ \(_ :: (KnownNat rows) => Proxy rows) ->
     reifyNat cols $ \(_ :: (KnownNat cols) => Proxy cols) ->
     reifyNat depth $ \(_ :: (KnownNat depth) => Proxy depth) ->
     case (rows, cols, depth) of
         (_, 1, 1)    -> return $ SpecLayer Gelu (sing :: Sing ('D1 rows)) (sing :: Sing ('D1 rows))
         (_, _, 1) -> return $ SpecLayer Gelu (sing :: Sing ('D2 rows cols)) (sing :: Sing ('D2 rows cols))
         _    -> case (unsafeCoerce (Dict :: Dict()) :: Dict (KnownNat (rows GHC.TypeLits.* depth))) of
           Dict -> return $ SpecLayer Gelu (sing :: Sing ('D3 rows cols depth)) (sing :: Sing ('D3 rows cols depth))


-- | Create a specification for a elu layer.
specGelu1D :: Integer -> SpecNet
specGelu1D i = specGelu3D (i, 1, 1)

-- | Create a specification for a elu layer.
specGelu2D :: (Integer, Integer) -> SpecNet
specGelu2D (i, j) = specGelu3D (i, j, 1)

-- | Create a specification for a elu layer.
specGelu3D :: (Integer, Integer, Integer) -> SpecNet
specGelu3D = SpecNetLayer . SpecGelu


-- | Add a Gelu layer to your build.
gelu :: BuildM ()
gelu = buildGetLastLayerOut >>= buildAddSpec . SpecNetLayer . SpecGelu


-------------------- GNum instances --------------------

instance GNum Gelu where
  _ |* Gelu = Gelu
  _ |+ Gelu = Gelu
  zipVectorsWithInPlaceReplSnd _ _ Gelu = Gelu
