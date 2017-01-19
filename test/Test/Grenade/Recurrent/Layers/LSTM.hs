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

import           Disorder.Jack

import           Data.Foldable ( toList )
import           Data.Singletons.TypeLits

import           Grenade
import           Grenade.Recurrent

import qualified Numeric.LinearAlgebra as H
import qualified Numeric.LinearAlgebra.Static as S


import qualified Test.Grenade.Recurrent.Layers.LSTM.Reference as Reference
import           Test.Jack.Hmatrix

genLSTM :: forall i o. (KnownNat i, KnownNat o) => Jack (LSTM i o)
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
  gamble randomVector $ \(input :: S.R 3) ->
    gamble randomVector $ \(cell :: S.R 2) ->
      gamble genLSTM $ \(net@(LSTM lstmWeights _) :: LSTM 3 2) ->
        let actual          = runRecurrentForwards net (S1D cell) (S1D input)
        in case actual of
          ((S1D cellOut) :: S ('D1 2), (S1D output) :: S ('D1 2)) ->
            let cellOut'        = Reference.Vector . H.toList . S.extract $ cellOut
                output'         = Reference.Vector . H.toList . S.extract $ output
                refNet          = Reference.lstmToReference lstmWeights
                refCell         = Reference.Vector . H.toList . S.extract $ cell
                refInput        = Reference.Vector . H.toList . S.extract $ input
                (refCO, refO)   = Reference.runLSTM refNet refCell refInput
            in toList refCO ~~~ toList cellOut' .&&. toList refO ~~~ toList output'

prop_lstm_reference_backwards =
  gamble randomVector $ \(input :: S.R 3) ->
    gamble randomVector $ \(cell :: S.R 2) ->
      gamble genLSTM $ \(net@(LSTM lstmWeights _) :: LSTM 3 2) ->
        let actualBacks     = runRecurrentBackwards net (S1D cell) (S1D input) (S1D (S.konst 1) :: S ('D1 2)) (S1D (S.konst 1) :: S ('D1 2))
        in case actualBacks of
          (actualGradients, _, _) ->
            let refNet          = Reference.lstmToReference lstmWeights
                refCell         = Reference.Vector . H.toList . S.extract $ cell
                refInput        = Reference.Vector . H.toList . S.extract $ input
                refGradients    = Reference.runLSTMback refCell refInput refNet
            in toList refGradients ~~~ toList (Reference.lstmToReference actualGradients)

prop_lstm_reference_backwards_input =
  gamble randomVector $ \(input :: S.R 3) ->
    gamble randomVector $ \(cell :: S.R 2) ->
      gamble genLSTM $ \(net@(LSTM lstmWeights _) :: LSTM 3 2) ->
        let actualBacks     = runRecurrentBackwards net (S1D cell) (S1D input) (S1D (S.konst 1) :: S ('D1 2)) (S1D (S.konst 1) :: S ('D1 2))
        in case actualBacks of
          (_, _, S1D actualGradients) ->
            let refNet          = Reference.lstmToReference lstmWeights
                refCell         = Reference.Vector . H.toList . S.extract $ cell
                refInput        = Reference.Vector . H.toList . S.extract $ input
                refGradients    = Reference.runLSTMbackOnInput refCell refNet refInput
            in toList refGradients ~~~ H.toList (S.extract actualGradients)

prop_lstm_reference_backwards_cell =
  gamble randomVector $ \(input :: S.R 3) ->
    gamble randomVector $ \(cell :: S.R 2) ->
      gamble genLSTM $ \(net@(LSTM lstmWeights _) :: LSTM 3 2) ->
        let actualBacks     = runRecurrentBackwards net (S1D cell) (S1D input) (S1D (S.konst 1) :: S ('D1 2)) (S1D (S.konst 1) :: S ('D1 2))
        in case actualBacks of
          (_, S1D actualGradients, _) ->
            let refNet          = Reference.lstmToReference lstmWeights
                refCell         = Reference.Vector . H.toList . S.extract $ cell
                refInput        = Reference.Vector . H.toList . S.extract $ input
                refGradients    = Reference.runLSTMbackOnCell refInput refNet refCell
            in toList refGradients ~~~ H.toList (S.extract actualGradients)


(~~~) as bs = all (< 1e-8) (zipWith (-) as bs)
infix 4 ~~~

return []
tests :: IO Bool
tests = $quickCheckAll
