{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE ScopedTypeVariables   #-}

module Test.Jack.Hmatrix where

import           Data.Proxy
import           Disorder.Jack

import           GHC.TypeLits

import qualified Numeric.LinearAlgebra.Static as HStatic

randomVector :: forall n. KnownNat n => Jack (HStatic.R n)
randomVector = HStatic.fromList <$> vectorOf (fromInteger (natVal (Proxy :: Proxy n))) sizedRealFrac

uniformSample :: forall m n. (KnownNat m, KnownNat n) => Jack (HStatic.L m n)
uniformSample = HStatic.fromList
             <$> vectorOf (fromInteger (natVal (Proxy :: Proxy m)) * fromInteger (natVal (Proxy :: Proxy n)))
                  sizedRealFrac
