{-# LANGUAGE CPP                   #-}
{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE PolyKinds             #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE TemplateHaskell       #-}
{-# LANGUAGE FlexibleContexts      #-}

module Test.Grenade.Network.Accelerate where

import qualified Prelude as P
import Prelude hiding (zipWith, replicate, length, maximum)

import           Data.Constraint
#if __GLASGOW_HASKELL__ < 800
import           Data.Proxy
#endif
import           Data.Singletons.Prelude.List
import           Data.Singletons
import           Hedgehog
import qualified Hedgehog.Gen as Gen

import Data.Array.Accelerate
import Data.Array.Accelerate.Interpreter

import Grenade as G
import Grenade.Core.Accelerate as GA
import Test.Grenade.Network ((~~~), maxVal, oneUp)

-- import           Data.Type.Equality

import           GHC.TypeLits
import           GHC.TypeLits.Witnesses

import           Test.Hedgehog.Compat
import           Test.Hedgehog.Hmatrix
import           Test.Hedgehog.TypeLits
import           Unsafe.Coerce


data SomeNetwork :: * where
  SomeNetwork ::
    (
      SingI shapes
    , SingI (Head shapes)
    , SingI (Last shapes)
    , Show (G.Network layers shapes)
    , Accelerable (G.Network layers shapes)
    , Accelerable (G.S (Head shapes))
    , Accelerable (G.S (Last shapes))
    ) => G.Network layers shapes -> SomeNetwork

instance Show SomeNetwork where
  show (SomeNetwork net) = show net


-- | Generate a random network of a random type
--
-- This is slightly insane for a few reasons. Everything must be wrapped up
-- in a SomeNetwork.
genNetwork :: Monad m => Gen.Gen m SomeNetwork
genNetwork =
  Gen.recursive Gen.choice [
    do
      output  :: Integer  <- choose 1 100
      let Just output' = someNatVal output
      case (output') of
        (SomeNat (Proxy :: Proxy o')) ->   do
          pure (SomeNetwork (G.NNil :: G.Network '[] '[ 'D1 o' ] ))
  ] [
    do
      SomeNetwork ( rest :: G.Network layers shapes ) <- genNetwork
      case ( sing :: Sing shapes ) of
        SNil -> Gen.discard -- Can't occur
        SCons ( h :: Sing h ) ( _ :: Sing hs ) ->
          withSingI h $
          case h of
            D1Sing l@SNat -> do -- Reshape to two dimensions
              let divisors n = 1 : [x | x <- [2..(n-1)], n `rem` x Prelude.== 0]
              let len = natVal l
              rs  <- Gen.element $ divisors len
              let cs  = len `quot` rs
              case ( someNatVal rs, someNatVal cs, someNatVal len ) of
                ( Just (SomeNat (rs' :: Proxy inRows)), Just (SomeNat (cs' :: Proxy inCols)), Just (SomeNat (_ :: Proxy outLen ) )) ->
                  let p1 = natDict rs'
                      p2 = natDict cs'
                  in  case ( p1 %* p2, unsafeCoerce (Dict :: Dict ()) :: Dict ((inRows * inCols) ~ outLen), unsafeCoerce (Dict :: Dict ()) :: Dict (( 'D1 outLen ) ~ h )) of
                        ( Dict, Dict, Dict ) -> do
                          wB    <- randomVector
                          bM    <- randomVector
                          wN    <- uniformSample
                          kM    <- uniformSample
                          let layer = FullyConnected (FullyConnected' wB wN) (FullyConnected' bM kM)
                          return (SomeNetwork (layer :~> rest :: G.Network ( FullyConnected inRows outLen ': layers ) ( ('D1 inRows) ': h ': hs )))
                _ -> Gen.discard -- Doesn't occur
            D2Sing r@SNat c@SNat -> Gen.discard
            D3Sing r@SNat c@SNat f@SNat -> Gen.discard
  ]


type AH s = Accelerated (G.S (Head s))
type AL s = Accelerated (G.S (Last s))


-- | Test a partial derivative numerically for a random network and input
--
-- This is the most important test.
prop_auto_diff :: Property
prop_auto_diff = withDiscards 1000 . withTests 10000 . property $ do
  SomeNetwork (network :: G.Network layers shapes) <- forAll genNetwork
  (input  :: G.S (Head shapes))     <- forAllWith nice genOfShape
  (target :: G.S (Last shapes))     <- forAllWith nice oneUp
  (tested :: G.S (Head shapes))     <- forAllWith nice oneUp

  let
    (!tapes, !output) = G.runNetwork network input
    (_, !backgrad)    = G.runGradient network tapes target
    inputDiff         = input + tested * 0.00001
    expected          = maxVal ( backgrad * tested )
    (_, !outputDiff)  = G.runNetwork network inputDiff
    result            = maxVal ( outputDiff * target - output * target ) / 0.00001

    networkA = toAccel network

    inputA :: AH shapes
    inputA = toAccel input

    targetA :: AL shapes
    targetA = toAccel target

    testedA :: AH shapes
    testedA = toAccel tested

    tapesA :: GA.Tapes layers shapes
    outputA :: AL shapes
    (!tapesA, !outputA) = GA.runNetwork networkA inputA

    backgradA :: AH shapes
    (_, !backgradA)     = GA.runGradient networkA tapesA targetA

    sA :: Exp DIM1
    sA = case inputA of
      AS1D v -> shape v

    inputDiffA :: AH shapes
    inputDiffA = _diffInputs inputA testedA sA

    outputDiffA :: AL shapes
    (_, !outputDiffA)   = GA.runNetwork networkA inputDiffA

    (result', expected') = run $ _results backgradA testedA targetA outputA outputDiffA

  result ~~~ expected
  result' ~~~ expected'

_diffInputs (AS1D inputV) (AS1D testedV) sA = AS1D $ zipWith (+) inputV (zipWith (*) testedV (fill sA 0.00001))

_results
  :: (Accelerated (G.S ('D1 i)))
  -> (Accelerated (G.S ('D1 i)))
  -> (Accelerated (G.S ('D1 o)))
  -> (Accelerated (G.S ('D1 o)))
  -> (Accelerated (G.S ('D1 o)))
  -> Acc (Double, Double)
_results (AS1D backgradV) (AS1D testedV) (AS1D targetV) (AS1D outputV) (AS1D outputDiffV) =
      let
        expectedA = maximum (zipWith (*) backgradV testedV)
        resultA = maximum (zipWith (-) (zipWith (*) outputDiffV targetV) (zipWith (*) outputV targetV)) / 0.00001
      in lift (resultA, expectedA)

tests :: IO Bool
tests = checkParallel $$(discover)
