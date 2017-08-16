{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FunctionalDependencies #-}

module Grenade.Core.Matrix.Accelerate where

import qualified Prelude as P
import           Data.Proxy
import           GHC.TypeLits

import           Data.Array.Accelerate hiding (flatten, size)
import           Data.Array.Accelerate.IO
import           Numeric.LinearAlgebra.Static (R, L, unwrap, size)
import           Numeric.LinearAlgebra (flatten)

class Accelerable g a | a -> g where
  -- | Accelerate a Grenade type
  toAccel :: g -> a


outer :: (P.Num (Exp e), Elt e) => Acc (Vector e) -> Acc (Vector e) -> Acc (Array DIM2 e)
outer a b = zipWith (*) aRepl bRepl
  where
    (Z :. aN) = unlift $ shape a :: Z :. Exp Int
    (Z :. bN) = unlift $ shape b :: Z :. Exp Int
    aRepl = replicate (lift $ Z :. All :. bN) a
    bRepl = replicate (lift $ Z :. aN :. All) b

(#>) :: (P.Num (Exp e), Elt e) => Acc (Array DIM2 e) -> Acc (Vector e) -> Acc (Vector e)
m #> v = fold (+) 0 mul
  where
    (Z :. vN) = unlift $ shape v :: Z :. Exp Int
    (Z :. mN :. nN) = unlift $ shape m :: Z :. Exp Int :. Exp Int

    vRepl = replicate (lift $ Z :. mN :. All) v
    mul = zipWith (*) m vRepl

fromMatrix :: (KnownNat m, KnownNat n) => L m n -> Array DIM2 Double
fromMatrix mat = (fromVectors (Z :. rows :. cols)) $ flatten $ unwrap mat
  where
    (rows, cols) = size mat

fromVector :: (KnownNat n) => R n -> Vector Double
fromVector vec = (fromVectors (Z :. len)) $ unwrap vec
  where
    len = size vec
