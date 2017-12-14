{-# LANGUAGE RankNTypes            #-}
module Test.Hedgehog.Compat (
    (...)
  , choose
  , blindForAll
  )where

import           Hedgehog (Gen)
import qualified Hedgehog.Gen as Gen
import qualified Hedgehog.Range as Range
import           Hedgehog.Internal.Property

(...) :: (c -> d) -> (a -> b -> c) -> a -> b -> d
(...) = (.) . (.)
{-# INLINE (...) #-}

choose :: Integral a => a -> a -> Gen a
choose = Gen.integral ... Range.constant

blindForAll :: Monad m => Gen a -> PropertyT m a
blindForAll = forAllWith (const "blind")
