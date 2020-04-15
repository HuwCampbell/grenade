{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE RankNTypes          #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies        #-}
{-# LANGUAGE TypeOperators       #-}

module Grenade.Recurrent.Core.Runner (
    runRecurrentExamples
  , runRecurrentBackprop
  , backPropagateRecurrent
  , trainRecurrent

  , RecurrentGradients
  ) where

import           Data.List                      (foldl')
import           Data.Singletons.Prelude
import           Grenade.Core

import           Grenade.Recurrent.Core.Network

type RecurrentGradients layers = [RecurrentGradient layers]

runRecurrentExamples  :: forall shapes layers.
                         RecurrentNetwork layers shapes
                      -> RecurrentInputs layers
                      -> [S (Head shapes)]
                      -> ([(RecurrentTape layers shapes, S (Last shapes))], RecurrentInputs layers)
runRecurrentExamples net =
  go
    where
  go !side [] = ([], side)
  go !side (!x : xs) =
    let (!tape, !side', !o) = runRecurrent net side x
        (!res, !finalSide)  = go side' xs
    in  (( tape, o ) : res, finalSide)

runRecurrentBackprop  :: forall layers shapes.
                         RecurrentNetwork layers shapes
                      -> RecurrentInputs layers
                      -> [(RecurrentTape layers shapes, S (Last shapes))]
                      -> ([(RecurrentGradient layers, S (Head shapes))], RecurrentInputs layers)
runRecurrentBackprop net =
  go
    where
  go !side [] = ([], side)
  go !side ((!tape,!x):xs) =
    let (res, !side')            = go side xs
        (!grad, !finalSide, !o)  = runRecurrent' net tape side' x
    in  (( grad, o ) : res, finalSide)


-- | Drive and network and collect its back propogated gradients.
backPropagateRecurrent :: forall shapes layers. (SingI (Last shapes), Fractional (RecurrentInputs layers))
                       => RecurrentNetwork layers shapes
                       -> RecurrentInputs layers
                       -> [(S (Head shapes), Maybe (S (Last shapes)))]
                       -> (RecurrentGradients layers, RecurrentInputs layers)
backPropagateRecurrent network recinputs examples =
  let (outForwards, _)       = runRecurrentExamples network recinputs inputs

      backPropagations       = zipWith makeError outForwards targets

      (outBackwards, input') = runRecurrentBackprop network 0 backPropagations

      gradients              = fmap fst outBackwards
  in (gradients, input')

    where

  inputs  = fst <$> examples
  targets = snd <$> examples

  makeError :: (x, S (Last shapes)) -> Maybe (S (Last shapes)) -> (x, S (Last shapes))
  makeError (x, _) Nothing  = (x, 0)
  makeError (x, y) (Just t) = (x, y - t)


trainRecurrent :: forall opt shapes layers. (SingI (Last shapes), Fractional (RecurrentInputs layers))
               => Optimizer opt
               -> RecurrentNetwork layers shapes
               -> RecurrentInputs layers
               -> [(S (Head shapes), Maybe (S (Last shapes)))]
               -> (RecurrentNetwork layers shapes, RecurrentInputs layers)
trainRecurrent opt network recinputs examples =
  let (gradients, recinputs') = backPropagateRecurrent network recinputs examples
      newInputs               = updateRecInputs opt recinputs recinputs'
      newNetwork              = foldl' (applyRecurrentUpdate opt) network gradients
  in  (newNetwork, newInputs)

updateRecInputs :: Optimizer opt -> RecurrentInputs sublayers -> RecurrentInputs sublayers -> RecurrentInputs sublayers
updateRecInputs opt (() :~~+> xs) (() :~~+> ys) = () :~~+> updateRecInputs opt xs ys
updateRecInputs opt@(OptSGD lRate _ lRegulariser) (x :~@+> xs) (y :~@+> ys) = (realToFrac (1 - lRate * lRegulariser) * x - realToFrac lRate * y) :~@+> updateRecInputs opt xs ys
updateRecInputs _ RINil RINil = RINil
