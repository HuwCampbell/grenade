{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE DeriveAnyClass        #-}
{-# LANGUAGE DeriveGeneric         #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TypeOperators         #-}
{-|
Module      : Grenade.Layers.LeakyRelu
Description : Rectifying linear unit layer
Copyright   : (c) Manuel Schneckenreither, 2020
License     : BSD2
Stability   : experimental
-}
module Grenade.Layers.LeakyRelu (
    LeakyRelu (..)
  , SpecLeakyRelu (..)
  , specLeakyRelu1D
  , specLeakyRelu2D
  , specLeakyRelu3D
  , leakyRelu
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
import           Grenade.Layers.Internal.BLAS   (toLayerShape)
import           Grenade.Types
import           Grenade.Utils.Vector


-- | A rectifying linear unit.
--   A layer which can act between any shape of the same dimension, acting as a
--   diode on every neuron individually.
data LeakyRelu = LeakyRelu
  deriving (Generic, NFData, Show)

instance UpdateLayer LeakyRelu where
  type Gradient LeakyRelu = ()
  runUpdate _ _ _ = LeakyRelu

instance RandomLayer LeakyRelu where
  createRandomWith _ _ = return LeakyRelu

instance Serialize LeakyRelu where
  put _ = return ()
  get = return LeakyRelu

instance (KnownNat i) => Layer LeakyRelu ('D1 i) ('D1 i) where
  type Tape LeakyRelu ('D1 i) ('D1 i) = S ('D1 i)

  runForwards _ (S1D y) = (S1D y, S1D (relu y))
    where
      relu = LAS.dvmap (\a -> if a < 0 then alpha * a else a)
  runForwards _ (S1DV y) = (S1DV y, S1DV (parMapVectorC c_leaky_relu y))
  runBackwards _ (S1D y) (S1D dEdy) = ((), S1D (relu' y * dEdy))
    where
      relu' = LAS.dvmap (\a -> if a < 0 then alpha else 1)
  runBackwards _ (S1DV y) (S1DV dEdy) = ((), S1DV $ parZipWithVectorReplSndC c_leaky_relu_dif_fast y dEdy)
  runBackwards x y dEdy = runBackwards x y (toLayerShape y dEdy)

instance (KnownNat i, KnownNat j) => Layer LeakyRelu ('D2 i j) ('D2 i j) where
  type Tape LeakyRelu ('D2 i j) ('D2 i j) = S ('D2 i j)

  runForwards _ (S2D y) = (S2D y, S2D (relu y))
    where
      relu = LAS.dmmap (\a -> if a < 0 then alpha * a else a)
  runForwards _ (S2DV y) = (S2DV y, S2DV (parMapVectorC c_leaky_relu y))
  runBackwards _ (S2D y) (S2D dEdy) = ((), S2D (relu' y * dEdy))
    where
      relu' = LAS.dmmap (\a -> if a < 0 then alpha else 1)
  runBackwards _ (S2DV y) (S2DV dEdy) = ((), S2DV $ parZipWithVectorReplSndC c_leaky_relu_dif_fast y dEdy)
  runBackwards x y dEdy = runBackwards x y (toLayerShape y dEdy)

instance (KnownNat i, KnownNat j, KnownNat k) => Layer LeakyRelu ('D3 i j k) ('D3 i j k) where

  type Tape LeakyRelu ('D3 i j k) ('D3 i j k) = S ('D3 i j k)

  runForwards _ (S3D y) = (S3D y, S3D (relu y))
    where
      relu = LAS.dmmap (\a -> if a < 0 then alpha * a else a)
  runBackwards _ (S3D y) (S3D dEdy) = ((), S3D (relu' y * dEdy))
    where
      relu' = LAS.dmmap (\a -> if a < 0 then alpha else 1)

alpha :: RealNum
alpha = 0.02


-------------------- DynamicNetwork instance --------------------

instance FromDynamicLayer LeakyRelu where
  fromDynamicLayer inp _ _ = SpecNetLayer $ SpecLeakyRelu (tripleFromSomeShape inp)

instance ToDynamicLayer SpecLeakyRelu where
  toDynamicLayer _ _ (SpecLeakyRelu (rows, cols, depth)) =
     reifyNat rows $ \(_ :: (KnownNat rows) => Proxy rows) ->
     reifyNat cols $ \(_ :: (KnownNat cols) => Proxy cols) ->
     reifyNat depth $ \(_ :: (KnownNat depth) => Proxy depth) ->
     case (rows, cols, depth) of
         (_, 1, 1)    -> return $ SpecLayer LeakyRelu (sing :: Sing ('D1 rows)) (sing :: Sing ('D1 rows))
         (_, _, 1) -> return $ SpecLayer LeakyRelu (sing :: Sing ('D2 rows cols)) (sing :: Sing ('D2 rows cols))
         _    -> case (unsafeCoerce (Dict :: Dict()) :: Dict (KnownNat (rows GHC.TypeLits.* depth))) of
           Dict -> return $ SpecLayer LeakyRelu (sing :: Sing ('D3 rows cols depth)) (sing :: Sing ('D3 rows cols depth))


-- | Create a specification for a elu layer.
specLeakyRelu1D :: Integer -> SpecNet
specLeakyRelu1D i = specLeakyRelu3D (i, 1, 1)

-- | Create a specification for a elu layer.
specLeakyRelu2D :: (Integer, Integer) -> SpecNet
specLeakyRelu2D (i, j) = specLeakyRelu3D (i, j, 1)

-- | Create a specification for a elu layer.
specLeakyRelu3D :: (Integer, Integer, Integer) -> SpecNet
specLeakyRelu3D = SpecNetLayer . SpecLeakyRelu


-- | Add a LeakyRelu layer to your build.
leakyRelu :: BuildM ()
leakyRelu = buildGetLastLayerOut >>= buildAddSpec . SpecNetLayer . SpecLeakyRelu


-------------------- GNum instances --------------------

instance GNum LeakyRelu where
  _ |* LeakyRelu = LeakyRelu
  _ |+ LeakyRelu = LeakyRelu
  gFromRational _ = LeakyRelu
