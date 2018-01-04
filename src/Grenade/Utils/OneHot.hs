{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE RankNTypes            #-}

module Grenade.Utils.OneHot (
    oneHot
  , hotMap
  , makeHot
  , unHot
  , sample
  ) where

import qualified Control.Monad.Random as MR

import           Data.List ( group, sort )

import           Data.Map ( Map )
import qualified Data.Map as M

import           Data.Proxy
import           Data.Singletons.TypeLits

import           Data.Vector ( Vector )
import qualified Data.Vector as V
import qualified Data.Vector.Storable as VS

import           Numeric.LinearAlgebra ( maxIndex )
import           Numeric.LinearAlgebra.Devel
import           Numeric.LinearAlgebra.Static

import           Grenade.Core.Shape

-- | From an int which is hot, create a 1D Shape
--   with one index hot (1) with the rest 0.
--   Rerurns Nothing if the hot number is larger
--   than the length of the vector.
oneHot :: forall n. (KnownNat n)
       => Int -> Maybe (S ('D1 n))
oneHot hot =
  let len = fromIntegral $ natVal (Proxy :: Proxy n)
  in if hot < len
      then
        fmap S1D . create $ runSTVector $ do
        vec    <- newVector 0 len
        writeVector vec hot 1
        return vec
      else Nothing

-- | Create a one hot map from any enumerable.
--   Returns a map, and the ordered list for the reverse transformation
hotMap :: (Ord a, KnownNat n) => Proxy n -> [a] -> Either String (Map a Int, Vector a)
hotMap n as =
  let len  = fromIntegral $ natVal n
      uniq = [ c | (c:_) <- group $ sort as]
      hotl = length uniq
  in if hotl == len
      then
        Right (M.fromList $ zip uniq [0..], V.fromList uniq)
      else
        Left ("Couldn't create hotMap of size " ++ show len ++ " from vector with " ++ show hotl ++ " unique characters")

-- | From a map and value, create a 1D Shape
--   with one index hot (1) with the rest 0.
--   Rerurns Nothing if the hot number is larger
--   than the length of the vector or the map
--   doesn't contain the value.
makeHot :: forall a n. (Ord a, KnownNat n)
        => Map a Int -> a -> Maybe (S ('D1 n))
makeHot m x = do
  hot    <- M.lookup x m
  let len = fromIntegral $ natVal (Proxy :: Proxy n)
  if hot < len
      then
        fmap S1D . create $ runSTVector $ do
        vec    <- newVector 0 len
        writeVector vec hot 1
        return vec
      else Nothing

unHot :: forall a n. KnownNat n
      => Vector a -> S ('D1 n) -> Maybe a
unHot v (S1D xs)
  = (V.!?) v
  $ maxIndex (extract xs)

sample :: forall a n m. (KnownNat n, MR.MonadRandom m)
       => Double -> Vector a -> S ('D1 n) -> m a
sample temperature v (S1D xs) = do
  ix <- MR.fromList . zip [0..] . fmap (toRational . exp . (/ temperature) . log) . VS.toList . extract $ xs
  return $ v V.! ix
