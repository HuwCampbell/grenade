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
import           Data.Constraint (Dict (..))
import           Data.Reflection (reifyNat)
import           Data.Serialize
import           Data.Singletons
import           GHC.Generics    (Generic)
import           GHC.TypeLits
import           Unsafe.Coerce   (unsafeCoerce)


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
  toDynamicLayer _ _ (SpecTrivial (rows, cols, depth)) =
     reifyNat rows $ \(_ :: (KnownNat rows) => Proxy rows) ->
     reifyNat cols $ \(_ :: (KnownNat cols) => Proxy cols) ->
     reifyNat depth $ \(_ :: (KnownNat depth) => Proxy depth) ->
     case (rows, cols, depth) of
         (_, 0, 0)    -> return $ SpecLayer Trivial (sing :: Sing ('D1 rows)) (sing :: Sing ('D1 rows))
         (_, _, 0) -> return $ SpecLayer Trivial (sing :: Sing ('D2 rows cols)) (sing :: Sing ('D2 rows cols))
         _    -> case (unsafeCoerce (Dict :: Dict()) :: Dict (KnownNat (rows GHC.TypeLits.* depth))) of
           Dict -> return $ SpecLayer Trivial (sing :: Sing ('D3 rows cols depth)) (sing :: Sing ('D3 rows cols depth))


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


