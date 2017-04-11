{-# LANGUAGE RankNTypes            #-}
module Test.Hedgehog.Compat where

import           Control.Monad.Trans.Class (MonadTrans(..))

import           Data.Typeable ( typeOf )

import           Hedgehog.Gen ( Gen )
import qualified Hedgehog.Gen as Gen
import qualified Hedgehog.Range as Range
import           Hedgehog.Internal.Property
import           Hedgehog.Internal.Source
import           Hedgehog.Internal.Show

(...) :: (c -> d) -> (a -> b -> c) -> a -> b -> d
(...) = (.) . (.)
{-# INLINE (...) #-}

choose :: ( Monad m, Integral a ) => a -> a -> Gen m a
choose = Gen.integral ... Range.constant

-- | Generates a random input for the test by running the provided generator.
blindForAll :: Monad m => Gen m a -> Test m a
blindForAll = Test . lift . lift

-- | Generates a random input for the test by running the provided generator.
semiBlindForAll :: (Monad m, Show a, HasCallStack) => Gen m a -> Test m a
semiBlindForAll gen = do
  x <- Test . lift $ lift gen
  writeLog $ Input (getCaller callStack) (typeOf ()) (showPretty x)
  return x


-- | Generates a random input for the test by running the provided generator.
forAllRender :: (Monad m, HasCallStack) => ( a -> String ) -> Gen m a -> Test m a
forAllRender render gen = do
  x <- Test . lift $ lift gen
  writeLog $ Input (getCaller callStack) (typeOf ()) (render x)
  return x
