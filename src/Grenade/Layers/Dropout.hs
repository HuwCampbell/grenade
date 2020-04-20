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
  , dropout
  , dropoutWithSeed
  ) where

import           Control.DeepSeq
import           Control.Monad.Primitive        (PrimBase, PrimState)
import           Data.Proxy
import           Data.Reflection                (reifyNat)
import           Data.Serialize
import           Data.Singletons
import           GHC.Generics                   hiding (R)
import           GHC.TypeLits
import           Numeric.LinearAlgebra.Static   hiding (Seed)
import           System.Random.MWC

import           Grenade.Core
import           Grenade.Dynamic
import           Grenade.Dynamic.Internal.Build
import           Grenade.Types

-- Dropout layer help to reduce overfitting.
-- Idea here is that the vector is a shape of 1s and 0s, which we multiply the input by.
-- After backpropogation, we return a new matrix/vector, with different bits dropped out.
-- The provided argument is the proportion to drop in each training iteration (like 1% or
-- 5% would be reasonable).
data Dropout (pct :: Nat) =
  Dropout
    { dropoutActive :: Bool     -- ^ Add possibility to deactivate dropout
    , dropoutSeed   :: !Int     -- ^ Seed
    }
  deriving (Generic)

instance NFData (Dropout pct) where rnf (Dropout a s) = rnf a `seq` rnf s
instance Show (Dropout pct) where show (Dropout _ _) = "Dropout"
instance Serialize (Dropout pct) where
  put (Dropout act seed) = put act >> put seed
  get = Dropout <$> get <*> get

instance UpdateLayer (Dropout pct) where
  type Gradient (Dropout pct) = ()
  runUpdate _ (Dropout act seed) _ = Dropout act (seed+1)
  runSettingsUpdate set (Dropout _ seed) = Dropout (setDropoutActive set) seed

instance RandomLayer (Dropout pct) where
  createRandomWith _ = randomDropout

randomDropout :: (PrimBase m) => Gen (PrimState m) -> m (Dropout pct)
randomDropout gen = Dropout True <$> uniform gen

instance (KnownNat pct, KnownNat i) => Layer (Dropout pct) ('D1 i) ('D1 i) where
  type Tape (Dropout pct) ('D1 i) ('D1 i) = R i
  runForwards (Dropout act seed) (S1D x)
    | not act = (v, S1D $ dvmap (rate *) x) -- multily with rate to normalise throughput
    | otherwise = (v, S1D $ v * x)
    where
      rate = (/100) $ fromIntegral $ max 0 $ min 100 $ natVal (Proxy :: Proxy pct)
      v = dvmap mask $ randomVector seed Uniform
      mask r
        | not act || r < rate = 1
        | otherwise = 0
  runBackwards (Dropout _ _) v (S1D x) = ((), S1D $ x * v)

-------------------- DynamicNetwork instance --------------------

instance (KnownNat pct) => FromDynamicLayer (Dropout pct) where
  fromDynamicLayer inp _ (Dropout _ seed) = case tripleFromSomeShape inp of
    (rows, 1, 1) -> SpecNetLayer $ SpecDropout rows rate (Just seed)
    _            -> error "Dropout is only allows for vectors, i.e. 1D spaces."
    where rate = (/100) $ fromIntegral $ max 0 $ min 100 $ natVal (Proxy :: Proxy pct)

instance ToDynamicLayer SpecDropout where
  toDynamicLayer _ gen (SpecDropout rows rate mSeed) =
    reifyNat rows $ \(_ :: (KnownNat i) => Proxy i) ->
    reifyNat (round $ 100 * rate) $ \(_ :: (KnownNat pct) => Proxy pct) ->
    case mSeed of
      Just seed -> return $ SpecLayer (Dropout True seed :: Dropout pct) (sing :: Sing ('D1 i)) (sing :: Sing ('D1 i))
      Nothing -> do
        layer <-  randomDropout gen
        return $ SpecLayer (layer :: Dropout pct) (sing :: Sing ('D1 i)) (sing :: Sing ('D1 i))


-- | Create a specification for a droput layer by providing the input size of the vector (1D allowed only!), a rate of nodes to keep (e.g. 0.95) and maybe a seed.
specDropout :: Integer -> RealNum -> Maybe Int -> SpecNet
specDropout i rate seed = SpecNetLayer $ SpecDropout i rate seed

-- | Create a dropout layer with the specified keep rate of nodes. The seed will be randomly initialized when the network is created. See also @dropoutWithSeed@.
dropout :: RealNum -> BuildM ()
dropout ratio = dropoutWithSeed ratio Nothing

-- | Create a dropout layer with the specified keep rate of nodes. The seed will be randomly initialized when the network is created. See also @dropoutWithSeed@.
dropoutWithSeed :: RealNum -> Maybe Int -> BuildM ()
dropoutWithSeed ratio mSeed = buildRequireLastLayerOut Is1D >>= \(i, _, _) -> buildAddSpec (specDropout i ratio mSeed)


-------------------- GNum instance --------------------

instance GNum (Dropout pct) where
  _ |* x = x
  _ |+ x = x
  gFromRational r = Dropout True (round r)
