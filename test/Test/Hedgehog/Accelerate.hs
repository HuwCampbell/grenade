{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE FlexibleContexts      #-}

module Test.Hedgehog.Accelerate where

import qualified Prelude as P
import           Prelude (Monad, (<$>))
import           Data.Array.Accelerate
import           Data.Array.Accelerate.Interpreter

import           Hedgehog
import qualified Hedgehog.Gen as Gen
import qualified Hedgehog.Range as Range

randomArray :: ( Monad m, Shape sh ) => sh -> Gen.Gen m (Array sh Double)
randomArray sh = fromList sh <$> Gen.list (Range.singleton $ arraySize sh) (Gen.realFloat $ Range.linearFracFrom 0 (-1) 1)

(~===) :: (P.Num (Exp e), P.Fractional (Exp e), RealFrac e, Monad m, P.Eq sh, P.Eq e, Elt e, Shape sh, FromIntegral Int e) => Array sh e -> Array sh e -> Test m ()
a ~=== b = fuzzy a === fuzzy b
  where
    fuzzy :: (P.Num (Exp e), P.Fractional (Exp e), RealFrac e, Shape sh, Elt e, FromIntegral Int e) => Array sh e -> Array sh e
    fuzzy = run1 $ map $ \x ->
      let
        scaledUp :: Exp Int
        scaledUp = round $ x * 1e7
      in (fromIntegral scaledUp) / 1e7
