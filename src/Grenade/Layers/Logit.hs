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
  , logit
  ) where


import           Control.DeepSeq                (NFData)
import           Data.Constraint                (Dict (..))
import           Data.Reflection
import           Data.Serialize
import           Data.Singletons
import           GHC.Generics                   (Generic)
import           GHC.TypeLits
import           Grenade.Layers.Internal.BLAS   (toLayerShape)
import           Unsafe.Coerce                  (unsafeCoerce)

import           Grenade.Core
import           Grenade.Dynamic
import           Grenade.Dynamic.Internal.Build
import           Grenade.Utils.Vector

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

instance (a ~ b, SingI a) => Layer Logit a b
  -- Wengert tape optimisation:
  --
  -- Derivative of the sigmoid function is
  --    d σ(x) / dx  = σ(x) • (1 - σ(x))
  -- but we have already calculated σ(x) in
  -- the forward pass, so just store that
  -- and use it in the backwards pass.
                                       where
  type Tape Logit a b = S a
  runForwards _ (S1DV vec) =
    let l = parMapVectorC c_sigmoid vec
     in (S1DV l, S1DV l)
  runForwards _ (S2DV vec) =
    let l = parMapVectorC c_sigmoid vec
     in (S2DV l, S2DV l)
  runForwards _ a =
    let l = sigmoid a
     in (l, l)
  runBackwards _ (S1DV vec) (S1DV g) = ((), S1DV $ parZipWithVectorReplSndC c_sigmoid_dif_fast vec g)
  runBackwards _ (S2DV vec) (S2DV g) = ((), S2DV $ parZipWithVectorReplSndC c_sigmoid_dif_fast vec g)
  runBackwards _ l g =
    let sigmoid' = l * (1 - l)
        g' = toLayerShape l g
     in ((), sigmoid' * g')

instance Serialize Logit where
  put _ = return ()
  get = return Logit

sigmoid :: Floating a => a -> a
sigmoid x = 1 / (1 + exp (-x))


-------------------- DynamicNetwork instance --------------------

instance FromDynamicLayer Logit where
  fromDynamicLayer inp _ _ = SpecNetLayer $ SpecLogit (tripleFromSomeShape inp)

instance ToDynamicLayer SpecLogit where
  toDynamicLayer _ _ (SpecLogit (rows, cols, depth)) =
     reifyNat rows $ \(_ :: (KnownNat rows) => Proxy rows) ->
     reifyNat cols $ \(_ :: (KnownNat cols) => Proxy cols) ->
     reifyNat depth $ \(_ :: (KnownNat depth) => Proxy depth) ->
     case (rows, cols, depth) of
         (_, 1, 1)    -> return $ SpecLayer Logit (sing :: Sing ('D1 rows)) (sing :: Sing ('D1 rows))
         (_, _, 1) -> return $ SpecLayer Logit (sing :: Sing ('D2 rows cols)) (sing :: Sing ('D2 rows cols))
         _    -> case (unsafeCoerce (Dict :: Dict()) :: Dict (KnownNat (rows GHC.TypeLits.* depth))) of
           Dict -> return $ SpecLayer Logit (sing :: Sing ('D3 rows cols depth)) (sing :: Sing ('D3 rows cols depth))


-- | Create a specification for a elu layer.
specLogit1D :: Integer -> SpecNet
specLogit1D i = specLogit3D (i, 1, 1)

-- | Create a specification for a elu layer.
specLogit2D :: (Integer, Integer) -> SpecNet
specLogit2D (i, j) = specLogit3D (i, j, 1)

-- | Create a specification for a elu layer.
specLogit3D :: (Integer, Integer, Integer) -> SpecNet
specLogit3D = SpecNetLayer . SpecLogit


-- | Add a Logit layer to your build.
logit :: BuildM ()
logit = buildGetLastLayerOut >>= buildAddSpec . SpecNetLayer . SpecLogit


-------------------- GNum instances --------------------

instance GNum Logit where
  _ |* x = x
  _ |+ x = x
  gFromRational _ = Logit
