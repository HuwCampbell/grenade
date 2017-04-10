{-# LANGUAGE RankNTypes            #-}
module Test.Jack.Compat where

import           Hedgehog
import qualified Hedgehog.Gen as Gen
import qualified Hedgehog.Range as Range

type Jack x = forall m. Monad m => Gen m x

(...) :: (c -> d) -> (a -> b -> c) -> a -> b -> d
(...) = (.) . (.)
{-# INLINE (...) #-}

choose :: Integral a => a -> a -> Jack a
choose = Gen.integral ... Range.constant

