{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE DeriveAnyClass        #-}
{-# LANGUAGE DeriveGeneric         #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TypeOperators         #-}
{-|
Module      : Grenade.Layers.Sinusoid
Description : Sinusoid nonlinear layer
Copyright   : (c) Manuel Schneckenreither, 2018
License     : BSD2
Stability   : experimental
-}
module Grenade.Layers.Sinusoid
  ( Sinusoid(..)
  , SpecSinusoid (..)
  , specSinusoid1D
  , specSinusoid2D
  , specSinusoid3D
  , sinusoid
  ) where

import           Control.DeepSeq                (NFData)
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

-- | A Sinusoid layer.
--   A layer which can act between any shape of the same dimension, performing a sin function.
data Sinusoid = Sinusoid
  deriving (NFData, Generic, Show)

instance UpdateLayer Sinusoid where
  type Gradient Sinusoid  = ()
  runUpdate _ _ _ = Sinusoid

instance RandomLayer Sinusoid where
  createRandomWith _ _ = return Sinusoid

instance Serialize Sinusoid where
  put _ = return ()
  get = return Sinusoid

instance (a ~ b, SingI a) => Layer Sinusoid a b where
  type Tape Sinusoid a b = S a
  runForwards _ a = (a, sin a)
  runBackwards _ a g = ((), cos a * g)


-------------------- DynamicNetwork instance --------------------

instance FromDynamicLayer Sinusoid where
  fromDynamicLayer inp _ _ = SpecNetLayer $ SpecSinusoid (tripleFromSomeShape inp)

instance ToDynamicLayer SpecSinusoid where
  toDynamicLayer _ _ (SpecSinusoid (rows, cols, depth)) =
     reifyNat rows $ \(_ :: (KnownNat rows) => Proxy rows) ->
     reifyNat cols $ \(_ :: (KnownNat cols) => Proxy cols) ->
     reifyNat depth $ \(_ :: (KnownNat depth) => Proxy depth) ->
     case (rows, cols, depth) of
         (_, 1, 1)    -> return $ SpecLayer Sinusoid (sing :: Sing ('D1 rows)) (sing :: Sing ('D1 rows))
         (_, _, 1) -> return $ SpecLayer Sinusoid (sing :: Sing ('D2 rows cols)) (sing :: Sing ('D2 rows cols))
         _    -> case (unsafeCoerce (Dict :: Dict()) :: Dict (KnownNat (rows GHC.TypeLits.* depth))) of
           Dict -> return $ SpecLayer Sinusoid (sing :: Sing ('D3 rows cols depth)) (sing :: Sing ('D3 rows cols depth))


-- | Create a specification for a elu layer.
specSinusoid1D :: Integer -> SpecNet
specSinusoid1D i = specSinusoid3D (i, 1, 1)

-- | Create a specification for a elu layer.
specSinusoid2D :: (Integer, Integer) -> SpecNet
specSinusoid2D (i,j) = specSinusoid3D (i,j,1)

-- | Create a specification for a elu layer.
specSinusoid3D :: (Integer, Integer, Integer) -> SpecNet
specSinusoid3D = SpecNetLayer . SpecSinusoid


-- | Add a Sinusoid layer to your build.
sinusoid :: BuildM ()
sinusoid = buildGetLastLayerOut >>= buildAddSpec . SpecNetLayer . SpecSinusoid


-------------------- GNum instances --------------------

instance GNum Sinusoid where
  _ |* Sinusoid = Sinusoid
  _ |+ Sinusoid = Sinusoid
  gFromRational _ = Sinusoid
