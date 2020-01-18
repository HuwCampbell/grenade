{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE KindSignatures        #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE ScopedTypeVariables   #-}

module Test.Hedgehog.Hmatrix where

import           Grenade
import           Data.Singletons
import           Data.Singletons.TypeLits

import           Hedgehog (Gen)
import qualified Hedgehog.Gen as Gen
import qualified Hedgehog.Range as Range

import qualified Numeric.LinearAlgebra.Static as HStatic

randomVector :: forall n. ( KnownNat n ) =>  Gen (HStatic.R n)
randomVector = (\s -> HStatic.randomVector s HStatic.Uniform * 2 - 1) <$> Gen.int Range.linearBounded

uniformSample :: forall m n. ( KnownNat m, KnownNat n ) => Gen (HStatic.L m n)
uniformSample = (\s -> HStatic.uniformSample s (-1) 1 ) <$> Gen.int Range.linearBounded

-- | Generate random data of the desired shape
genOfShape :: forall x. ( SingI x ) => Gen (S x)
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
