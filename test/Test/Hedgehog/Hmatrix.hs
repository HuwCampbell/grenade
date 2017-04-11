{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE GADTs                 #-}

module Test.Hedgehog.Hmatrix where

import           Grenade
import           Data.Singletons

import           Hedgehog.Gen ( Gen )
import qualified Hedgehog.Gen as Gen
import qualified Hedgehog.Range as Range

import           GHC.TypeLits

import qualified Numeric.LinearAlgebra.Static as HStatic

randomVector :: forall m n. ( Monad m, KnownNat n ) =>  Gen m (HStatic.R n)
randomVector = (\s -> HStatic.randomVector s HStatic.Uniform * 2 - 1) <$> Gen.int Range.linearBounded

uniformSample :: forall mm m n. ( Monad mm, KnownNat m, KnownNat n ) => Gen mm (HStatic.L m n)
uniformSample = (\s -> HStatic.uniformSample s (-1) 1 ) <$> Gen.int Range.linearBounded

-- | Generate random data of the desired shape
genOfShape :: forall m x. ( Monad m, SingI x ) =>  Gen m (S x)
genOfShape =
  case (sing :: Sing x) of
    D1Sing -> S1D <$> randomVector
    D2Sing -> S2D <$> uniformSample
    D3Sing -> S3D <$> uniformSample


nice :: S shape -> String
nice (S1D x) = show . HStatic.extract $ x
nice (S2D x) = show . HStatic.extract $ x
nice (S3D x) = show . HStatic.extract $ x
