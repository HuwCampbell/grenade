{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE DeriveAnyClass        #-}
{-# LANGUAGE DeriveGeneric         #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE Strict                #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TypeOperators         #-}
{-|
Module      : Grenade.Layers.LeakyTanh
Description : Hyperbolic tangent nonlinear layer
Copyright   : (c) Huw Campbell, 2016-2017
License     : BSD2
Stability   : experimental
-}
module Grenade.Layers.LeakyTanh
  ( LeakyTanh(..)
  , SpecLeakyTanh (..)
  , specLeakyTanh1D
  , specLeakyTanh2D
  , specLeakyTanh3D
  , specLeakyTanh
  , leakyTanhLayer
  ) where

import           Control.DeepSeq                (NFData (..))
import           Data.Constraint                (Dict (..))
import           Data.Proxy
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
import           Grenade.Types
import           Grenade.Utils.Conversion
import           Grenade.Utils.Vector

-- | A LeakyTanh layer. A layer which can act between any shape of the same dimension, performing a tanh function. The maximum value is given in Promille, i.e. 995 is 0.995, and is positive and negative. I.e. LeakyTanh(x) = min 0.995 (max (-0.995) x).
data LeakyTanh (v :: Nat) = LeakyTanh
  deriving (Generic,NFData,Show)

instance UpdateLayer (LeakyTanh maxVal) where
  type Gradient (LeakyTanh maxVal) = ()
  runUpdate _ l@LeakyTanh{} _ = l

instance RandomLayer (LeakyTanh maxVal) where
  createRandomWith _ _ = return LeakyTanh


instance Serialize (LeakyTanh maxVal) where
  put _ = return ()
  get = return LeakyTanh

instance (a ~ b, SingI a, KnownNat maxVal) => Layer (LeakyTanh maxVal) a b where
  type Tape (LeakyTanh maxVal) a b = S a
  runForwards _ (S1DV v) = (S1DV v, S1DV $ mapVector (tanhMax maxVal) v)
    where
      maxVal = (/ 1000) $ fromIntegral $ max 0 $ min 1000 $ natVal (Proxy :: Proxy maxVal)
  runForwards _ (S2DV v) = (S2DV v, S2DV $ mapVector (tanhMax maxVal) v)
    where
      maxVal = (/ 1000) $ fromIntegral $ max 0 $ min 1000 $ natVal (Proxy :: Proxy maxVal)
  runForwards _ (S1D a) = (S1D a, S1D $ LAS.dvmap (tanhMax maxVal) a)
    where
      maxVal = (/ 1000) $ fromIntegral $ max 0 $ min 1000 $ natVal (Proxy :: Proxy maxVal)
  runForwards _ (S2D a) = (S2D a, S2D $ LAS.dmmap (tanhMax maxVal) a)
    where
      maxVal = (/ 1000) $ fromIntegral $ max 0 $ min 1000 $ natVal (Proxy :: Proxy maxVal)
  runForwards _ (S3D a) = (S3D a, S3D $ LAS.dmmap (tanhMax maxVal) a)
    where
      maxVal = (/ 1000) $ fromIntegral $ max 0 $ min 1000 $ natVal (Proxy :: Proxy maxVal)
  runBackwards _ (S1DV v) (S1DV gs) = ((), S1DV $ zipWithVector (\t g -> tanhMax' maxVal t * g) v gs)
    where
      maxVal = (/ 1000) $ fromIntegral $ max 0 $ min 1000 $ natVal (Proxy :: Proxy maxVal)
  runBackwards _ (S2DV v) (S2DV gs) = ((), S2DV $ zipWithVector (\t g -> tanhMax' maxVal t * g) v gs)
    where
      maxVal = (/ 1000) $ fromIntegral $ max 0 $ min 1000 $ natVal (Proxy :: Proxy maxVal)
  runBackwards _ (S1D a) (S1D g) = ((), S1D $ LAS.dvmap (tanhMax' maxVal) a * g)
    where
      maxVal = (/ 1000) $ fromIntegral $ max 0 $ min 1000 $ natVal (Proxy :: Proxy maxVal)
  runBackwards _ (S2D a) (S2D g) = ((), S2D $ LAS.dmmap (tanhMax' maxVal) (tanh' a) * g)
    where
      maxVal = (/ 1000) $ fromIntegral $ max 0 $ min 1000 $ natVal (Proxy :: Proxy maxVal)
  runBackwards _ (S3D a) (S3D g) = ((), S3D $ LAS.dmmap (tanhMax' maxVal) (tanh' a) * g)
    where
      maxVal = (/ 1000) $ fromIntegral $ max 0 $ min 1000 $ natVal (Proxy :: Proxy maxVal)
  runBackwards l x y = runBackwards l x (toLayerShape x y)

tanhMax :: (Ord a, Fractional a) => a -> a -> a
tanhMax m v
  | v > m = leaky * (v - m) + m
  | v < -m = leaky * (v - m) - m
  | otherwise = v
  where
    leaky = 0.05

tanhMax' :: (Ord a, Fractional a, Floating a) => a -> a -> a
tanhMax' m v
  | v > m = leaky
  | v < -m = -leaky
  | otherwise = max 0.005 (tanh' v)
  where
    leaky = 0.05


tanh' :: (Floating a) => a -> a
tanh' t = 1 - s ^ (2 :: Int)  where s = tanh t

-------------------- DynamicNetwork instance --------------------

instance (KnownNat maxVal) => FromDynamicLayer (LeakyTanh maxVal) where
  fromDynamicLayer inp _ LeakyTanh = SpecNetLayer $ SpecLeakyTanh maxVal (tripleFromSomeShape inp)
    where maxVal = (/1000) $ fromIntegral $ max 0 $ min 1000 $ natVal (Proxy :: Proxy maxVal)

instance ToDynamicLayer SpecLeakyTanh where
  toDynamicLayer _ _ (SpecLeakyTanh maxVal (rows, cols, depth)) =
     reifyNat rows $ \(_ :: (KnownNat rows) => Proxy rows) ->
     reifyNat cols $ \(_ :: (KnownNat cols) => Proxy cols) ->
     reifyNat depth $ \(_ :: (KnownNat depth) => Proxy depth) ->
     reifyNat (round $ 1000 * maxVal) $ \(_ :: (KnownNat maxVal) => Proxy maxVal) ->
     case (rows, cols, depth) of
         (_, 1, 1) -> case (unsafeCoerce (Dict :: Dict()) :: Dict ()) of
           Dict -> return $ SpecLayer (LeakyTanh :: LeakyTanh maxVal) (sing :: Sing ('D1 rows)) (sing :: Sing ('D1 rows))
         (_, _, 1) -> case (unsafeCoerce (Dict :: Dict()) :: Dict ()) of
           Dict -> return $ SpecLayer (LeakyTanh :: LeakyTanh maxVal) (sing :: Sing ('D2 rows cols)) (sing :: Sing ('D2 rows cols))
         _         -> case (unsafeCoerce (Dict :: Dict()) :: Dict (KnownNat (rows GHC.TypeLits.* depth))) of
           Dict -> return $ SpecLayer (LeakyTanh :: LeakyTanh maxVal) (sing :: Sing ('D3 rows cols depth)) (sing :: Sing ('D3 rows cols depth))


-- | Create a specification for a LeakyTanh layer.
specLeakyTanh1D :: RealNum -> Integer -> SpecNet
specLeakyTanh1D maxVal i = specLeakyTanh3D maxVal (i, 1, 1)

-- | Create a specification for a LeakyTanh layer.
specLeakyTanh2D :: RealNum -> (Integer, Integer) -> SpecNet
specLeakyTanh2D maxVal (i, j) = specLeakyTanh3D maxVal (i, j, 1)

-- | Create a specification for a LeakyTanh layer.
specLeakyTanh3D :: RealNum -> (Integer, Integer, Integer) -> SpecNet
specLeakyTanh3D maxVal = SpecNetLayer . SpecLeakyTanh maxVal

-- | Create a specification for a LeakyTanh layer.
specLeakyTanh :: RealNum -> (Integer, Integer, Integer) -> SpecNet
specLeakyTanh maxVal = SpecNetLayer . SpecLeakyTanh maxVal

-- | Add a LeakyTanh layer to your build.
leakyTanhLayer :: RealNum -> BuildM ()
leakyTanhLayer maxVal
  | maxVal <= 0 || maxVal > 1 = error "The maxVal parameter of tanhMaxLayer has to be in (0,1]"
  | otherwise = buildGetLastLayerOut >>= buildAddSpec . SpecNetLayer . SpecLeakyTanh maxVal


-------------------- GNum instances --------------------

instance GNum (LeakyTanh maxVal) where
  _ |* _ = LeakyTanh
  _ |+ _ = LeakyTanh
  sumG _ = LeakyTanh
