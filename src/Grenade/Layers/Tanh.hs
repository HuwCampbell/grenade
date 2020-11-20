{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE DeriveAnyClass        #-}
{-# LANGUAGE DeriveGeneric         #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE Strict                #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TypeOperators         #-}
{-|
Module      : Grenade.Layers.Tanh
Description : Hyperbolic tangent nonlinear layer
Copyright   : (c) Huw Campbell, 2016-2017
License     : BSD2
Stability   : experimental
-}
module Grenade.Layers.Tanh
  ( Tanh(..)
  , SpecTanh (..)
  , specTanh1D
  , specTanh2D
  , specTanh3D
  , specTanh
  , tanhLayer
  ) where

import           Control.DeepSeq                (NFData (..))
import           Data.Constraint                (Dict (..))
import           Data.Reflection                (reifyNat)
import           Data.Serialize
import           Data.Singletons
import           GHC.Generics                   (Generic)
import           GHC.TypeLits
import           Unsafe.Coerce                  (unsafeCoerce)

import           Grenade.Core
import           Grenade.Dynamic
import           Grenade.Dynamic.Internal.Build
import           Grenade.Utils.Vector

-- | A Tanh layer.
--   A layer which can act between any shape of the same dimension, performing a tanh function.
data Tanh = Tanh
  deriving (Generic,NFData,Show)

instance UpdateLayer Tanh where
  type Gradient Tanh = ()
  runUpdate _ _ _ = Tanh

instance RandomLayer Tanh where
  createRandomWith _ _ = return Tanh

instance Serialize Tanh where
  put _ = return ()
  get = return Tanh

instance (a ~ b, SingI a) => Layer Tanh a b where
  type Tape Tanh a b = S a
  -- runForwards _ (S1DV v) = (S1DV v, S1DV $ mapVectorInPlace tanh v) -- This is inplace replacement!
  -- runForwards _ (S2DV v) = (S2DV v, S2DV $ mapVectorInPlace tanh v) -- This is inplace replacement!
  runForwards _ a        = (a, tanh a)
  -- runBackwards _ (S1DV v) (S1DV gs) = ((), S1DV $ zipWithVectorInPlaceSnd (\t g -> 1 - t ^ (2 :: Int) * g) v gs)
  -- runBackwards _ (S2DV v) (S2DV gs) = ((), S2DV $ zipWithVectorInPlaceSnd (\t g -> 1 - t ^ (2 :: Int) * g) v gs)
  runBackwards _ a g = ((), tanh' a * g)

tanh' :: (Floating a) => a -> a
tanh' t = 1 - s ^ (2 :: Int)  where s = tanh t

-------------------- DynamicNetwork instance --------------------

instance FromDynamicLayer Tanh where
  fromDynamicLayer inp _ _ = SpecNetLayer $ SpecTanh (tripleFromSomeShape inp)

instance ToDynamicLayer SpecTanh where
  toDynamicLayer _ _ (SpecTanh (rows, cols, depth)) =
     reifyNat rows $ \(_ :: (KnownNat rows) => Proxy rows) ->
     reifyNat cols $ \(_ :: (KnownNat cols) => Proxy cols) ->
     reifyNat depth $ \(_ :: (KnownNat depth) => Proxy depth) ->
     case (rows, cols, depth) of
         (_, 1, 1)    -> return $ SpecLayer Tanh (sing :: Sing ('D1 rows)) (sing :: Sing ('D1 rows))
         (_, _, 1) -> return $ SpecLayer Tanh (sing :: Sing ('D2 rows cols)) (sing :: Sing ('D2 rows cols))
         _    -> case (unsafeCoerce (Dict :: Dict()) :: Dict (KnownNat (rows GHC.TypeLits.* depth))) of
           Dict -> return $ SpecLayer Tanh (sing :: Sing ('D3 rows cols depth)) (sing :: Sing ('D3 rows cols depth))


-- | Create a specification for a Tanh layer.
specTanh1D :: Integer -> SpecNet
specTanh1D i = specTanh3D (i, 1, 1)

-- | Create a specification for a Tanh layer.
specTanh2D :: (Integer, Integer) -> SpecNet
specTanh2D (i, j) = specTanh3D (i, j, 1)

-- | Create a specification for a Tanh layer.
specTanh3D :: (Integer, Integer, Integer) -> SpecNet
specTanh3D = SpecNetLayer . SpecTanh

-- | Create a specification for a Tanh layer.
specTanh :: (Integer, Integer, Integer) -> SpecNet
specTanh = SpecNetLayer . SpecTanh

-- | Add a Tanh layer to your build.
tanhLayer :: BuildM ()
tanhLayer = buildGetLastLayerOut >>= buildAddSpec . SpecNetLayer . SpecTanh


-------------------- GNum instances --------------------

instance GNum Tanh where
  _ |* Tanh = Tanh
  _ |+ Tanh = Tanh
  zipVectorsWithInPlaceReplSnd _ _ Tanh = Tanh
  sumG _ = Tanh
