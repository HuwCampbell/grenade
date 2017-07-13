{-# LANGUAGE RankNTypes            #-}
module Test.Hedgehog.Compat (
    (...)
  , choose
  , blindForAll
  , semiBlindForAll
  , forAllRender
  )where

import           Control.Monad.Trans.Class (MonadTrans(..))

import qualified Hedgehog.Gen as Gen
import qualified Hedgehog.Range as Range
import           Hedgehog.Internal.Property
import           Hedgehog.Internal.Source
import           Hedgehog.Internal.Show

(...) :: (c -> d) -> (a -> b -> c) -> a -> b -> d
(...) = (.) . (.)
{-# INLINE (...) #-}

choose :: ( Monad m, Integral a ) => a -> a -> Gen.Gen m a
choose = Gen.integral ... Range.constant

blindForAll :: Monad m => Gen.Gen m a -> Test m a
blindForAll = Test . lift . lift

semiBlindForAll :: (Monad m, Show a, HasCallStack) => Gen.Gen m a -> Test m a
semiBlindForAll gen = do
  x <- Test . lift $ lift gen
  withFrozenCallStack $ annotate (showPretty x)
  return x

forAllRender :: (Monad m, HasCallStack) => ( a -> String ) -> Gen.Gen m a -> Test m a
forAllRender render gen = do
  x <- Test . lift $ lift gen
  withFrozenCallStack $ footnote (render x)
  return x
