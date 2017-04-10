{-# LANGUAGE RankNTypes            #-}
module Test.Jack.Compat where

import           Control.Monad.Trans.Class (MonadTrans(..))

import           Hedgehog
import qualified Hedgehog.Gen as Gen
import qualified Hedgehog.Range as Range
import           Hedgehog.Internal.Property ( Test (..) )

type Jack x = forall m. Monad m => Gen m x

(...) :: (c -> d) -> (a -> b -> c) -> a -> b -> d
(...) = (.) . (.)
{-# INLINE (...) #-}

choose :: Integral a => a -> a -> Jack a
choose = Gen.integral ... Range.constant

-- | Generates a random input for the test by running the provided generator.
blindForAll :: Monad m => Gen m a -> Test m a
blindForAll = Test . lift . lift
