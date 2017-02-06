{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE CPP                   #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE RecordWildCards       #-}

#if __GLASGOW_HASKELL__ < 800
{-# OPTIONS_GHC -fno-warn-incomplete-patterns #-}
#endif

module Grenade.Recurrent.Core.Runner (
    trainRecurrent
  , runRecurrent
  , backPropagateRecurrent
  ) where

import           Data.Singletons.Prelude
import           Grenade.Core

import           Grenade.Recurrent.Core.Layer
import           Grenade.Recurrent.Core.Network

-- | Drive and network and collect its back propogated gradients.
backPropagateRecurrent :: forall shapes layers. (SingI (Last shapes), Num (RecurrentInputs layers))
                       => RecurrentNetwork layers shapes
                       -> RecurrentInputs layers
                       -> [(S (Head shapes), Maybe (S (Last shapes)))]
                       -> (RecurrentGradients layers, RecurrentInputs layers)
backPropagateRecurrent network recinputs examples =
  let (tapes, _, guesses)    = runRecurrentNetwork network recinputs inputs

      backPropagations       = zipWith makeError guesses targets

      (gradients, input', _) = runRecurrentGradient network tapes 0 backPropagations

  in (gradients, input')

    where

  inputs  = fst <$> examples
  targets = snd <$> examples

  makeError :: S (Last shapes) -> Maybe (S (Last shapes)) -> S (Last shapes)
  makeError _ Nothing = 0
  makeError y (Just t) = y - t


trainRecurrent :: forall shapes layers. (SingI (Last shapes), Num (RecurrentInputs layers))
               => LearningParameters
               -> RecurrentNetwork layers shapes
               -> RecurrentInputs layers
               -> [(S (Head shapes), Maybe (S (Last shapes)))]
               -> (RecurrentNetwork layers shapes, RecurrentInputs layers)
trainRecurrent rate network recinputs examples =
  let (gradients, recinputs') = backPropagateRecurrent network recinputs examples

      newInputs               = updateRecInputs rate recinputs recinputs'

      newNetwork              = applyRecurrentUpdate rate network gradients

  in  (newNetwork, newInputs)

updateRecInputs :: LearningParameters
                -> RecurrentInputs sublayers
                -> RecurrentInputs sublayers
                -> RecurrentInputs sublayers

updateRecInputs l@LearningParameters {..} (() :~~+> xs) (() :~~+> ys)
  = () :~~+> updateRecInputs l xs ys

updateRecInputs l@LearningParameters {..} (x :~@+> xs) (y :~@+> ys)
  = (realToFrac (learningRate * learningRegulariser) * x - realToFrac learningRate * y) :~@+> updateRecInputs l xs ys

updateRecInputs _ RINil RINil
  = RINil

-- | Just forwards propagation with no training.
runRecurrent :: RecurrentNetwork layers shapes
             -> RecurrentInputs layers -> S (Head shapes)
             -> (RecurrentInputs layers, S (Last shapes))
runRecurrent (layer :~~> n) (()    :~~+> nr) !x
  = let (_, ys)  = runForwards layer x
        (nr', o) = runRecurrent n nr ys
    in  (() :~~+> nr', o)
runRecurrent (layer :~@> n) (recin :~@+> nr) !x
  = let (_, recin', y) = runRecurrentForwards layer recin x
        (nr', o)       = runRecurrent n nr y
    in  (recin' :~@+> nr', o)
runRecurrent RNil RINil !x
  = (RINil, x)
