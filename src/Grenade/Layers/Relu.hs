{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE DeriveAnyClass        #-}
{-# LANGUAGE DeriveGeneric         #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE Strict                #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TypeOperators         #-}
{-|
Module      : Grenade.Layers.Relu
Description : Rectifying linear unit layer
Copyright   : (c) Huw Campbell, 2016-2017
License     : BSD2
Stability   : experimental
-}
module Grenade.Layers.Relu (
    Relu (..)
  , SpecRelu (..)
  , specRelu1D
  , specRelu2D
  , specRelu3D
  , relu
  ) where

import           Control.Parallel.Strategies
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
import           Grenade.Utils.Conversion
import           Grenade.Utils.Vector


-- | A rectifying linear unit.
--   A layer which can act between any shape of the same dimension, acting as a
--   diode on every neuron individually.
data Relu = Relu
  deriving (Generic, NFData, Show)

instance UpdateLayer Relu where
  type Gradient Relu = ()
  runUpdate _ _ _ = Relu

instance RandomLayer Relu where
  createRandomWith _ _ = return Relu

instance Serialize Relu where
  put _ = return ()
  get = return Relu

-- Run forward 1D
runForward1D :: (KnownNat i) => Relu -> S ('D1 i) -> (Tape Relu ('D1 i) ('D1 i), S ('D1 i))
runForward1D _ (S1D y) = (S1D y, S1D (relu y))
    where
      relu = LAS.dvmap (\a -> if a <= 0 then 0 else a)
runForward1D _ (S1DV y) =
  (S1DV y, S1DV (mapVectorInPlace relu y)) -- y will be replaced, which however still leads to a correct implementation!
  -- force (mapVectorInPlace relu y) `seq` (S1DV y, S1DV y) -- y will be replaced, which however still leads to a correct implementation!
  -- (S1DV y, S1DV $ mapVector relu y) -- y will be replaced, which however still leads to a correct implementation!
    where
      relu a = if a <= 0 then 0 else a


reluDifZip :: (Ord a, Floating a) => a -> a -> a
reluDifZip a g = if a <= 0 then 0 else g

-- Run backward 1D
runBackward1D :: (KnownNat i) => Relu -> S ('D1 i) -> S ('D1 i) -> (Gradient Relu, S ('D1 i))
runBackward1D _ (S1D y) (S1D dEdy) = ((), S1D (relu' y * dEdy))
    where
      relu' = LAS.dvmap (\a -> if a <= 0 then 0 else 1)
runBackward1D _ (S1DV y) (S1DV dEdy) = ((), S1DV (zipWithVectorInPlaceSnd reluDifZip y dEdy))
runBackward1D l y dEdy = runBackward1D l y (toLayerShape y dEdy)


instance (KnownNat i) => Layer Relu ('D1 i) ('D1 i) where
  type Tape Relu ('D1 i) ('D1 i) = S ('D1 i)
  runForwards = runForward1D
  runBackwards = runBackward1D


runBackward2D :: (KnownNat i, KnownNat j) => Relu -> S ('D2 i j) -> S ('D2 i j) -> (Gradient Relu, S ('D2 i j))
runBackward2D _ (S2D y) (S2D dEdy) = ((), S2D (relu' y * dEdy))
    where
      relu' = LAS.dmmap (\a -> if a <= 0 then 0 else 1)
runBackward2D _ (S2DV y) (S2DV dEdy) = ((), S2DV (zipWithVectorInPlaceSnd reluDifZip y dEdy))
runBackward2D l y dEdy = runBackward2D l y (toLayerShape y dEdy)

instance (KnownNat i, KnownNat j) => Layer Relu ('D2 i j) ('D2 i j) where
  type Tape Relu ('D2 i j) ('D2 i j) = S ('D2 i j)

  runForwards _ (S2D y) = (S2D y, S2D (relu y))
    where
      relu = LAS.dmmap (\a -> if a <= 0 then 0 else a)
  runForwards _ (S2DV y) = (S2DV y, S2DV $ mapVectorInPlace relu y ) -- This replaces the y vector, but that doesn't harm the updatesx
    where
      relu a = if a <= 0 then 0 else a
  runBackwards = runBackward2D

instance (KnownNat i, KnownNat j, KnownNat k) => Layer Relu ('D3 i j k) ('D3 i j k) where

  type Tape Relu ('D3 i j k) ('D3 i j k) = S ('D3 i j k)

  runForwards _ (S3D y) = (S3D y, S3D (relu y))
    where
      relu = LAS.dmmap (\a -> if a <= 0 then 0 else a)
  runBackwards _ (S3D y) (S3D dEdy) = ((), S3D (relu' y * dEdy))
    where
      relu' = LAS.dmmap (\a -> if a <= 0 then 0 else 1)


-------------------- DynamicNetwork instance --------------------

instance FromDynamicLayer Relu where
  fromDynamicLayer inp _ _ = SpecNetLayer $ SpecRelu (tripleFromSomeShape inp)

instance ToDynamicLayer SpecRelu where
  toDynamicLayer _ _ (SpecRelu (rows, cols, depth)) =
     reifyNat rows $ \(_ :: (KnownNat rows) => Proxy rows) ->
     reifyNat cols $ \(_ :: (KnownNat cols) => Proxy cols) ->
     reifyNat depth $ \(_ :: (KnownNat depth) => Proxy depth) ->
     case (rows, cols, depth) of
         (_, 1, 1)    -> return $ SpecLayer Relu (sing :: Sing ('D1 rows)) (sing :: Sing ('D1 rows))
         (_, _, 1) -> return $ SpecLayer Relu (sing :: Sing ('D2 rows cols)) (sing :: Sing ('D2 rows cols))
         _    -> case (unsafeCoerce (Dict :: Dict()) :: Dict (KnownNat (rows GHC.TypeLits.* depth))) of
           Dict -> return $ SpecLayer Relu (sing :: Sing ('D3 rows cols depth)) (sing :: Sing ('D3 rows cols depth))


-- | Create a specification for a elu layer.
specRelu1D :: Integer -> SpecNet
specRelu1D i = specRelu3D (i, 1, 1)

-- | Create a specification for a elu layer.
specRelu2D :: (Integer, Integer) -> SpecNet
specRelu2D (i, j) = specRelu3D (i, j, 1)

-- | Create a specification for a elu layer.
specRelu3D :: (Integer, Integer, Integer) -> SpecNet
specRelu3D = SpecNetLayer . SpecRelu


-- | Add a Relu layer to your build.
relu :: BuildM ()
relu = buildGetLastLayerOut >>= buildAddSpec . SpecNetLayer . SpecRelu


-------------------- GNum instances --------------------

instance GNum Relu where
  _ |* Relu = Relu
  _ |+ Relu = Relu
  zipVectorsWithInPlaceReplSnd _ _ Relu = Relu
  sumG _ = Relu
