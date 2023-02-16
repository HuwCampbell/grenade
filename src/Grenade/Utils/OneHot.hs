{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE RankNTypes          #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies        #-}
{-# LANGUAGE TypeOperators       #-}

module Grenade.Utils.OneHot (
    oneHot
  , hotMap
  , makeHot
  , unHot
  , sample
  ) where

import           Data.List                    (group, sort)
import           Data.Map                     (Map)
import qualified Data.Map                     as M
import           Data.Proxy
import           Data.Vector                  (Vector)
import qualified Data.Vector                  as V
import qualified Data.Vector.Storable         as VS
import           GHC.TypeLits
import           Numeric.LinearAlgebra        (maxIndex)
import           Numeric.LinearAlgebra.Devel
import           Numeric.LinearAlgebra.Static
import           System.Random.MWC            hiding (create)

import           Grenade.Core.Shape
import           Grenade.Types


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

sample :: forall a n . (KnownNat n)
       => RealNum -> Vector a -> S ('D1 n) -> IO a
sample temperature v (S1D xs) = do
  ix <- randFromList . zip [0..] . fmap (toRational . exp . (/ temperature) . log) . VS.toList . extract $ xs
  return $ v V.! ix


-- | Sample a random value from a weighted list.  The total weight of all
-- elements must not be 0.
randFromList :: [(a,Rational)] -> IO a
randFromList [] = error "OneHot.randFromList called with empty list"
randFromList [(x,_)] = return x
randFromList xs | sumxs == 0 = error "OneHot.randFromList sum of weights was 0"
                | otherwise = do
                    r <- toRational <$> (withSystemRandom . asGenST $ \gen -> uniformR (0, fromRational sumxs :: RealNum) gen)
                    return . fst . head $ dropWhile ((< r) . snd) cs
  where sumxs = sum (map snd xs)
        cs = scanl1 (\(_,q) (y,s') -> (y, s'+q)) xs -- cumulative weight
