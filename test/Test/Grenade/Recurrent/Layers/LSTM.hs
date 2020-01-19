{-# LANGUAGE TemplateHaskell       #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE ConstraintKinds       #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE RankNTypes            #-}

{-# OPTIONS_GHC -fno-warn-missing-signatures #-}
module Test.Grenade.Recurrent.Layers.LSTM where

import           Hedgehog
import           Hedgehog.Internal.Source
import           Hedgehog.Internal.Show
import           Hedgehog.Internal.Property ( failWith, Diff (..) )

import           Data.Foldable ( toList )
import           Data.Singletons.TypeLits

import           Grenade
import           Grenade.Recurrent

import qualified Numeric.LinearAlgebra as H
import qualified Numeric.LinearAlgebra.Static as S


import qualified Test.Grenade.Recurrent.Layers.LSTM.Reference as Reference
import           Test.Hedgehog.Hmatrix

genLSTM :: forall i o. (KnownNat i, KnownNat o) => Gen (LSTM i o)
genLSTM = do
    let w = uniformSample
        u = uniformSample
        v = randomVector

        w0 = S.konst 0
        u0 = S.konst 0
        v0 = S.konst 0

    LSTM <$> (LSTMWeights <$> w <*> u <*> v <*> w <*> u <*> v <*> w <*> u <*> v <*> w <*> v)
         <*> pure (LSTMWeights w0 u0 v0 w0 u0 v0 w0 u0 v0 w0 v0)

prop_lstm_reference_forwards =
  property $ do
    input :: S.R 3                       <- forAll randomVector
    cell :: S.R 2                        <- forAll randomVector
    net@(LSTM lstmWeights _) :: LSTM 3 2 <- forAll genLSTM

    let actual          = runRecurrentForwards net (S1D cell) (S1D input)
    case actual of
      (_, (S1D cellOut) :: S ('D1 2), (S1D output) :: S ('D1 2)) ->
        let cellOut'        = Reference.Vector . H.toList . S.extract $ cellOut
            output'         = Reference.Vector . H.toList . S.extract $ output
            refNet          = Reference.lstmToReference lstmWeights
            refCell         = Reference.Vector . H.toList . S.extract $ cell
            refInput        = Reference.Vector . H.toList . S.extract $ input
            (refCO, refO)   = Reference.runLSTM refNet refCell refInput
        in do toList refCO ~~~ toList cellOut'
              toList refO ~~~ toList output'


prop_lstm_reference_backwards =
  property $ do
    input :: S.R 3                       <- forAll randomVector
    cell :: S.R 2                        <- forAll randomVector
    net@(LSTM lstmWeights _) :: LSTM 3 2 <- forAll genLSTM
    let (tape, _ :: S ('D1 2), _ :: S ('D1 2))
                                          = runRecurrentForwards  net (S1D cell) (S1D input)
        actualBacks                       = runRecurrentBackwards net tape (S1D (S.konst 1) :: S ('D1 2)) (S1D (S.konst 1) :: S ('D1 2))
    case actualBacks of
      (actualGradients, _, _ :: S ('D1 3)) ->
        let refNet          = Reference.lstmToReference lstmWeights
            refCell         = Reference.Vector . H.toList . S.extract $ cell
            refInput        = Reference.Vector . H.toList . S.extract $ input
            refGradients    = Reference.runLSTMback refCell refInput refNet
        in toList refGradients ~~~ toList (Reference.lstmToReference actualGradients)

prop_lstm_reference_backwards_input =
  property $ do
    input :: S.R 3                       <- forAll randomVector
    cell :: S.R 2                        <- forAll randomVector
    net@(LSTM lstmWeights _) :: LSTM 3 2 <- forAll genLSTM
    let (tape, _ :: S ('D1 2), _ :: S ('D1 2))
                                          = runRecurrentForwards  net (S1D cell) (S1D input)
        actualBacks                       = runRecurrentBackwards net tape (S1D (S.konst 1) :: S ('D1 2)) (S1D (S.konst 1) :: S ('D1 2))
    case actualBacks of
      (_, _, S1D actualGradients :: S ('D1 3)) ->
        let refNet          = Reference.lstmToReference lstmWeights
            refCell         = Reference.Vector . H.toList . S.extract $ cell
            refInput        = Reference.Vector . H.toList . S.extract $ input
            refGradients    = Reference.runLSTMbackOnInput refCell refNet refInput
        in toList refGradients ~~~ H.toList (S.extract actualGradients)

prop_lstm_reference_backwards_cell =
  property $ do
    input :: S.R 3                       <- forAll randomVector
    cell :: S.R 2                        <- forAll randomVector
    net@(LSTM lstmWeights _) :: LSTM 3 2 <- forAll genLSTM
    let (tape, _ :: S ('D1 2), _ :: S ('D1 2))
                                          = runRecurrentForwards  net (S1D cell) (S1D input)
        actualBacks                       = runRecurrentBackwards net tape (S1D (S.konst 1) :: S ('D1 2)) (S1D (S.konst 1) :: S ('D1 2))
    case actualBacks of
      (_, S1D actualGradients, _ :: S ('D1 3)) ->
        let refNet          = Reference.lstmToReference lstmWeights
            refCell         = Reference.Vector . H.toList . S.extract $ cell
            refInput        = Reference.Vector . H.toList . S.extract $ input
            refGradients    = Reference.runLSTMbackOnCell refInput refNet refCell
        in toList refGradients ~~~ H.toList (S.extract actualGradients)

(~~~) :: (Monad m, Eq a, Ord a, Num a, Fractional a, Show a, HasCallStack) => [a] -> [a] -> PropertyT m ()
(~~~) x y =
  if all (< 1e-8) (zipWith (-) x y) then
    success
  else
    case valueDiff <$> mkValue x <*> mkValue y of
      Nothing ->
        withFrozenCallStack $
          failWith Nothing $ unlines [
              "━━━ Not Simliar ━━━"
            , showPretty x
            , showPretty y
            ]
      Just differ ->
        withFrozenCallStack $
          failWith (Just $ Diff "Failed (" "- lhs" "~/~" "+ rhs" ")" differ) ""
infix 4 ~~~

tests :: IO Bool
tests = checkParallel $$(discover)
