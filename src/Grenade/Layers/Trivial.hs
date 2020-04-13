{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE DeriveAnyClass        #-}
{-# LANGUAGE DeriveGeneric         #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TypeOperators         #-}
{-|
Module      : Grenade.Core.Trivial
Description : Trivial layer which perfoms no operations on the data
Copyright   : (c) Huw Campbell, 2016-2017
License     : BSD2
Stability   : experimental
-}
module Grenade.Layers.Trivial
  ( Trivial(..)
  , SpecTrivial(..)
  , specTrivial1D
  , specTrivial2D
  , specTrivial3D
  ) where

import           Control.DeepSeq (NFData (..))
import           Data.Serialize
import           GHC.Generics    (Generic)

import           Grenade.Core


-- | A Trivial layer.
--
--   This can be used to pass an unchanged value up one side of a
--   graph, for a Residual network for example.
data Trivial = Trivial
  deriving (Generic,NFData,Show)

instance Serialize Trivial where
  put _ = return ()
  get = return Trivial

instance UpdateLayer Trivial where
  type Gradient Trivial = ()
  runUpdate _ _ _ = Trivial

instance RandomLayer Trivial where
  createRandomWith _ _ = return Trivial

instance (a ~ b) => Layer Trivial a b where
  type Tape Trivial a b = ()
  runForwards _ a = ((), a)
  runBackwards _ _ y = ((), y)

-------------------- DynamicNetwork instance --------------------

instance FromDynamicLayer Trivial where
  fromDynamicLayer inp _ = SpecNetLayer $ SpecTrivial (tripleFromSomeShape inp)

instance ToDynamicLayer SpecTrivial where
  toDynamicLayer _ _ (SpecTrivial inp) = mkToDynamicLayerForActiviationFunction Trivial inp

-- | Create a specification for a elu layer.
specTrivial1D :: Integer -> SpecNet
specTrivial1D i = specTrivial3D (i, 0, 0)

-- | Create a specification for a elu layer.
specTrivial2D :: (Integer, Integer) -> SpecNet
specTrivial2D (i,j) = specTrivial3D (i,j,0)

-- | Create a specification for a elu layer.
specTrivial3D :: (Integer, Integer, Integer) -> SpecNet
specTrivial3D = SpecNetLayer . SpecTrivial

-------------------- GNum instances --------------------

instance GNum Trivial where
  _ |* Trivial = Trivial
  _ |+ Trivial  = Trivial
  gFromRational _ = Trivial


