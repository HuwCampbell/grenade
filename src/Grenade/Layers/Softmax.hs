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
Module      : Grenade.Core.Softmax
Description : Softmax loss layer
Copyright   : (c) Huw Campbell, 2016-2017
License     : BSD2
Stability   : experimental
-}
module Grenade.Layers.Softmax (
    Softmax (..)
  , softmax
  , softmax'
  , SpecSoftmax (..)
  , specSoftmax
  , softmaxLayer
  ) where

import           Data.Serialize

import           Control.DeepSeq                (NFData (..))
import           Data.Reflection                (reifyNat)
import           Data.Singletons
import           GHC.Generics                   (Generic)
import           GHC.TypeLits
import           Numeric.LinearAlgebra.Static   as LAS

import           Grenade.Core
import           Grenade.Dynamic
import           Grenade.Dynamic.Internal.Build

-- | A Softmax layer
--
--   This layer is like a logit layer, but normalises
--   a set of matricies to be probabilities.
--
--   One can use this layer as the last layer in a network
--   if they need normalised probabilities.
data Softmax = Softmax
  deriving (Show, Generic, NFData)

instance UpdateLayer Softmax where
  type Gradient Softmax = ()
  runUpdate _ _ _ = Softmax

instance RandomLayer Softmax where
  createRandomWith _ _ = return Softmax

instance ( KnownNat i ) => Layer Softmax ('D1 i) ('D1 i) where
  type Tape Softmax ('D1 i) ('D1 i) = S ('D1 i)

  runForwards _ (S1D y) = (S1D y, S1D (softmax y))
  runBackwards _ (S1D y) (S1D dEdy) = ((), S1D (softmax' y dEdy))

instance Serialize Softmax where
  put _ = return ()
  get = return Softmax

softmax :: KnownNat i => LAS.R i -> LAS.R i
softmax xs =
  let xs' = LAS.dvmap exp xs
      s   = LAS.dot xs' 1
  in  LAS.dvmap (/ s) xs'

softmax' :: KnownNat i => LAS.R i -> LAS.R i -> LAS.R i
softmax' x grad =
  let yTy = outer sm sm
      d   = diag sm
      g   = d - yTy
  in  g #> grad
    where
  sm = softmax x

-------------------- DynamicNetwork instance --------------------

instance FromDynamicLayer Softmax where
  fromDynamicLayer inp _ Softmax = case tripleFromSomeShape inp of
    (rows, 1, 1) -> SpecNetLayer $ SpecSoftmax rows
    _ -> error "Error in specification: The layer Softmax may only be used with 1D input!"

instance ToDynamicLayer SpecSoftmax where
  toDynamicLayer _ _ (SpecSoftmax rows) =
    reifyNat rows $ \(_ :: (KnownNat i) => Proxy i) ->
    return $ SpecLayer Softmax (sing :: Sing ('D1 i)) (sing :: Sing ('D1 i))


-- | Create a specification for a elu layer.
specSoftmax :: Integer -> SpecNet
specSoftmax = SpecNetLayer . SpecSoftmax


-- | Add a Softmax layer to your build.
softmaxLayer :: BuildM ()
softmaxLayer = buildRequireLastLayerOut Is1D >>= buildAddSpec . SpecNetLayer . SpecSoftmax . fst3
  where
    fst3 (x, _, _) = x

-------------------- GNum instances --------------------


instance GNum Softmax where
  _ |* Softmax = Softmax
  _ |+ Softmax = Softmax
  zipVectorsWithInPlaceReplSnd _ _ Softmax = Softmax
