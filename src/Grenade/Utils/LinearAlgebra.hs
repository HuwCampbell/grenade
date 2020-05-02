{-# LANGUAGE DataKinds #-}
module Grenade.Utils.LinearAlgebra
    ( sumV
    , sumM
    , squareV
    , squareM
    ) where

import           GHC.TypeLits
import           Grenade.Types
import           Numeric.LinearAlgebra.Static


sumV :: (KnownNat n) => R n -> RealNum
sumV v = fromDoubleToRealNum $ v <.> 1

sumM :: (KnownNat m, KnownNat n) => L m n -> RealNum
sumM m = fromDoubleToRealNum $ (m #> 1) <.> 1


squareV :: (KnownNat n) => R n -> R n
squareV v = dvmap (^ (2 :: Int)) v

squareM :: (KnownNat m, KnownNat n) => L m n ->  L m n
squareM m = dmmap (^ (2 :: Int)) m


-- v :: R 10
-- v = vector [1..10]


-- m :: L 3 3
-- m = matrix [-3..5]
