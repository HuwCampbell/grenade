{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE DeriveAnyClass        #-}
{-# LANGUAGE DeriveGeneric         #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TypeOperators         #-}
{-|
Module      : Grenade.Layers.Logit
Description : Exponential linear unit layer
Copyright   : (c) Huw Campbell, 2016-2017
License     : BSD2
Stability   : experimental
-}
module Grenade.Layers.Elu
  ( Elu(..)
  , SpecElu (..)
  , specElu1D
  , specElu2D
  , specElu3D
  , elu
  ) where

import           Control.DeepSeq                (NFData)
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

-- | An exponential linear unit.
--   A layer which can act between any shape of the same dimension, acting as a
--   diode on every neuron individually.
data Elu = Elu
  deriving (Generic, NFData, Show)

instance UpdateLayer Elu where
  type Gradient Elu = ()
  runUpdate _ _ _ = Elu

instance RandomLayer Elu where
  createRandomWith _ _ = return Elu

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

-------------------- DynamicNetwork instance --------------------

instance FromDynamicLayer Elu where
  fromDynamicLayer inp _ _ = SpecNetLayer $ SpecElu (tripleFromSomeShape inp)

instance ToDynamicLayer SpecElu where
  toDynamicLayer _ _ (SpecElu (rows, cols, depth)) =
     reifyNat rows $ \(_ :: (KnownNat rows) => Proxy rows) ->
     reifyNat cols $ \(_ :: (KnownNat cols) => Proxy cols) ->
     reifyNat depth $ \(_ :: (KnownNat depth) => Proxy depth) ->
     case (rows, cols, depth) of
         (_, 1, 1)    -> return $ SpecLayer Elu (sing :: Sing ('D1 rows)) (sing :: Sing ('D1 rows))
         (_, _, 1) -> return $ SpecLayer Elu (sing :: Sing ('D2 rows cols)) (sing :: Sing ('D2 rows cols))
         _    -> case (unsafeCoerce (Dict :: Dict()) :: Dict (KnownNat (rows GHC.TypeLits.* depth))) of
           Dict -> return $ SpecLayer Elu (sing :: Sing ('D3 rows cols depth)) (sing :: Sing ('D3 rows cols depth))


-- | Create a specification for a elu layer.
specElu1D :: Integer -> SpecNet
specElu1D i = specElu3D (i, 1, 1)

-- | Create a specification for a elu layer.
specElu2D :: (Integer, Integer) -> SpecNet
specElu2D (i,j) = specElu3D (i,j,1)

-- | Create a specification for a elu layer.
specElu3D :: (Integer, Integer, Integer) -> SpecNet
specElu3D = SpecNetLayer . SpecElu

-- | Add a Elu layer to your build.
elu :: BuildM ()
elu = buildGetLastLayerOut >>= buildAddSpec . SpecNetLayer . SpecElu


-------------------- GNum instances --------------------


instance GNum Elu where
  _ |* Elu = Elu
  _ |+ Elu = Elu
