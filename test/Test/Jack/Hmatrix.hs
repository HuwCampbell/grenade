{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE GADTs                 #-}

module Test.Jack.Hmatrix where

import           Grenade
import           Data.Singletons

import           Disorder.Jack

import           GHC.TypeLits

import qualified Numeric.LinearAlgebra.Static as HStatic

randomVector :: forall n. KnownNat n => Jack (HStatic.R n)
randomVector = (\s -> HStatic.randomVector s HStatic.Uniform * 2 - 1) <$> sizedNat

uniformSample :: forall m n. (KnownNat m, KnownNat n) => Jack (HStatic.L m n)
uniformSample = (\s -> HStatic.uniformSample s (-1) 1 ) <$> sizedNat

-- | Generate random data of the desired shape
genOfShape :: forall x. ( SingI x ) => Jack (S x)
genOfShape =
  case (sing :: Sing x) of
    D1Sing -> S1D <$> randomVector
    D2Sing -> S2D <$> uniformSample
    D3Sing -> S3D <$> uniformSample
