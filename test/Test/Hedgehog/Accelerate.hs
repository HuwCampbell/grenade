{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE GADTs                 #-}

module Test.Hedgehog.Accelerate where

import           Data.Singletons
import           Data.Singletons.TypeLits
import           Data.Array.Accelerate

import qualified Hedgehog.Gen as Gen
import qualified Hedgehog.Range as Range

randomArray :: ( Monad m, Shape sh ) => sh -> Gen.Gen m (Array sh Double)
randomArray sh = fromList sh <$> Gen.list (Range.singleton $ arraySize sh) (Gen.realFloat $ Range.linearFracFrom 0 (-1) 1)
