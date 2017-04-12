{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE GADTs                 #-}

module Test.Hedgehog.Hmatrix where

import           Grenade
import           Data.Singletons
import           Data.Singletons.TypeLits

import qualified Hedgehog.Gen as Gen
import qualified Hedgehog.Range as Range

import qualified Numeric.LinearAlgebra.Static as HStatic

randomVector :: forall m n. ( Monad m, KnownNat n ) =>  Gen.Gen m (HStatic.R n)
randomVector = (\s -> HStatic.randomVector s HStatic.Uniform * 2 - 1) <$> Gen.int Range.linearBounded

uniformSample :: forall mm m n. ( Monad mm, KnownNat m, KnownNat n ) => Gen.Gen mm (HStatic.L m n)
uniformSample = (\s -> HStatic.uniformSample s (-1) 1 ) <$> Gen.int Range.linearBounded

-- | Generate random data of the desired shape
genOfShape :: forall m x. ( Monad m, SingI x ) =>  Gen.Gen m (S x)
genOfShape =
  case (sing :: Sing x) of
    D1Sing l ->
      withKnownNat l $
        S1D <$> randomVector
    D2Sing r c ->
      withKnownNat r $ withKnownNat c $
        S2D <$> uniformSample
    D3Sing r c d ->
      withKnownNat r $ withKnownNat c $ withKnownNat d $
        S3D <$> uniformSample

nice :: S shape -> String
nice (S1D x) = show . HStatic.extract $ x
nice (S2D x) = show . HStatic.extract $ x
nice (S3D x) = show . HStatic.extract $ x
