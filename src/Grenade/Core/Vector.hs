{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE KindSignatures        #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE StandaloneDeriving    #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE PolyKinds             #-}

module Grenade.Core.Vector (
    Vector
  , vectorZip
  , vecToList
  , mkVector
  ) where

import           Data.Proxy
import           GHC.TypeLits

-- | A more specific Tagged type, ensuring that a list
--   is equal to the Nat value.
newtype Vector (n :: Nat) a = Vector [a]

instance Foldable (Vector n) where
  foldr f b (Vector as) = foldr f b as

instance KnownNat n => Traversable (Vector n) where
  traverse f (Vector as) = fmap mkVector $ traverse f as

instance Functor (Vector n) where
  fmap f (Vector as) = Vector (fmap f as)

instance Show a => Show (Vector n a) where
  showsPrec d = showsPrec d . vecToList

instance Eq a => Eq (Vector n a) where
  (Vector as) == (Vector bs) = as == bs

mkVector :: forall n a. KnownNat n => [a] -> Vector n a
mkVector as
 = let du = fromIntegral . natVal $ (undefined :: Proxy n)
       la = length as
   in if (du == la)
        then Vector as
        else error $ "Error creating staticly sized Vector of length: " ++
                     show du ++ " list is of length:" ++ show la

vecToList :: Vector n a -> [a]
vecToList (Vector as) = as

vectorZip :: (a -> b -> c) -> Vector n a -> Vector n b -> Vector n c
vectorZip f (Vector as) (Vector bs) = Vector (zipWith f as bs)
