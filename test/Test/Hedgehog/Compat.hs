{-# LANGUAGE RankNTypes            #-}
module Test.Hedgehog.Compat (
    (...)
  , choose
  , blindForAll
  , semiBlindForAll
  , forAllRender
  )where

import           Hedgehog (Gen)
import qualified Hedgehog.Gen as Gen
import qualified Hedgehog.Range as Range
import           Hedgehog.Internal.Property
import           Hedgehog.Internal.Source

(...) :: (c -> d) -> (a -> b -> c) -> a -> b -> d
(...) = (.) . (.)
{-# INLINE (...) #-}

choose :: ( Integral a ) => a -> a -> Gen a
choose = Gen.integral ... Range.constant

blindForAll :: Monad m => Gen a -> PropertyT m a
blindForAll = forAllWith (const "blind")

semiBlindForAll :: (Monad m, Show a, HasCallStack) => Gen a -> PropertyT m a
semiBlindForAll = forAllWith (const "blind")

forAllRender :: (Monad m, HasCallStack) => ( a -> String ) -> Gen a -> PropertyT m a
forAllRender render gen = forAllWith render gen
