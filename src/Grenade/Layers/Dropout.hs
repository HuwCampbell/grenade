{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE DeriveAnyClass        #-}
{-# LANGUAGE DeriveGeneric         #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TypeOperators         #-}
module Grenade.Layers.Dropout (
    Dropout (..)
  , randomDropout
  , SpecDropout (..)
  , specDropout
  ) where

import           Control.DeepSeq
import           Control.Monad.Primitive (PrimBase, PrimState)
import           Data.Reflection         (reifyNat)
import           Data.Serialize
import           Data.Singletons
import           System.Random.MWC
import           GHC.Generics
import           GHC.TypeLits

import           Grenade.Core
import           Grenade.Types

-- Dropout layer help to reduce overfitting.
-- Idea here is that the vector is a shape of 1s and 0s, which we multiply the input by.
-- After backpropogation, we return a new matrix/vector, with different bits dropped out.
-- F is the proportion to drop in each training iteration (like 1% or 5% would be
-- reasonable).
data Dropout = Dropout {
    dropoutRate :: !F
  , dropoutSeed :: !Int
  } deriving (Generic, NFData, Show, Serialize)


instance UpdateLayer Dropout where
  type Gradient Dropout = ()
  runUpdate _ x _ = x

instance RandomLayer Dropout where
  createRandomWith _ = randomDropout 0.95

randomDropout :: PrimBase m
              => F -> Gen (PrimState m) -> m Dropout
randomDropout rate gen = Dropout rate <$> uniform gen

instance (KnownNat i) => Layer Dropout ('D1 i) ('D1 i) where
  type Tape Dropout ('D1 i) ('D1 i) = ()
  runForwards (Dropout _ _) (S1D x) = ((), S1D x)
  runBackwards (Dropout _ _) _ (S1D x) = ((),  S1D x)

-------------------- DynamicNetwork instance --------------------

instance FromDynamicLayer Dropout where
  fromDynamicLayer inp _ (Dropout rate seed) = case tripleFromSomeShape inp of
    (rows, 0, 0) -> SpecNetLayer $ SpecDropout rows rate (Just seed)
    _ -> error "Error in specification: The layer Dropout may only be used with 1D input!"

instance ToDynamicLayer SpecDropout where
  toDynamicLayer _ gen (SpecDropout rows rate mSeed) =
    reifyNat rows $ \(_ :: (KnownNat i) => Proxy i) ->
    case mSeed of
      Just seed -> return $ SpecLayer (Dropout rate seed) (sing :: Sing ('D1 i)) (sing :: Sing ('D1 i))
      Nothing -> do
        layer <-  randomDropout rate gen
        return $ SpecLayer layer (sing :: Sing ('D1 i)) (sing :: Sing ('D1 i))


-- | Create a specification for a droput layer by providing the input size of the vector (1D allowed only!), a rate of dropout (default: 0.95) and maybe a seed.
specDropout :: Integer -> F -> Maybe Int -> SpecNet
specDropout i rate seed = SpecNetLayer $ SpecDropout i rate seed


-------------------- GNum instance --------------------

instance GNum Dropout where
  _ |* x = x
  _ |+ x = x
  gFromRational r = Dropout 0.95 (round r)

