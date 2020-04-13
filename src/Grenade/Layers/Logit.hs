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
Module      : Grenade.Layers.Logit
Description : Sigmoid nonlinear layer
Copyright   : (c) Huw Campbell, 2016-2017
License     : BSD2
Stability   : experimental
-}
module Grenade.Layers.Logit (
    Logit (..)
  , SpecLogit (..)
  , specLogit1D
  , specLogit2D
  , specLogit3D
  ) where


import           Control.DeepSeq (NFData)
import           Data.Constraint (Dict (..))
import           Data.Reflection
import           Data.Serialize
import           Data.Singletons
import           GHC.Generics    (Generic)
import           GHC.TypeLits
import           Unsafe.Coerce   (unsafeCoerce)

import           Grenade.Core

-- | A Logit layer.
--
--   A layer which can act between any shape of the same dimension, perfoming an sigmoid function.
--   This layer should be used as the output layer of a network for logistic regression (classification)
--   problems.
data Logit = Logit
  deriving (Generic,NFData,Show)

instance UpdateLayer Logit where
  type Gradient Logit = ()
  runUpdate _ _ _ = Logit

instance RandomLayer Logit where
  createRandomWith _ _ = return Logit

instance (a ~ b, SingI a) => Layer Logit a b where
  -- Wengert tape optimisation:
  --
  -- Derivative of the sigmoid function is
  --    d σ(x) / dx  = σ(x) • (1 - σ(x))
  -- but we have already calculated σ(x) in
  -- the forward pass, so just store that
  -- and use it in the backwards pass.
  type Tape Logit a b = S a
  runForwards _ a =
    let l = sigmoid a
    in  (l, l)
  runBackwards _ l g =
    let sigmoid' = l * (1 - l)
    in  ((), sigmoid' * g)

instance Serialize Logit where
  put _ = return ()
  get = return Logit

sigmoid :: Floating a => a -> a
sigmoid x = 1 / (1 + exp (-x))


-------------------- DynamicNetwork instance --------------------

instance FromDynamicLayer Logit where
  fromDynamicLayer inp _ = SpecNetLayer $ SpecLogit (tripleFromSomeShape inp)

instance ToDynamicLayer SpecLogit where
  toDynamicLayer _ _ (SpecLogit inp) = mkToDynamicLayerForActiviationFunction Logit inp


-- | Create a specification for a elu layer.
specLogit1D :: Integer -> SpecNet
specLogit1D i = specLogit3D (i, 0, 0)

-- | Create a specification for a elu layer.
specLogit2D :: (Integer, Integer) -> SpecNet
specLogit2D (i,j) = specLogit3D (i,j,0)

-- | Create a specification for a elu layer.
specLogit3D :: (Integer, Integer, Integer) -> SpecNet
specLogit3D = SpecNetLayer . SpecLogit


-------------------- GNum instances --------------------

instance GNum Logit where
  _ |* x = x
  _ |+ x = x
  gFromRational _ = Logit
