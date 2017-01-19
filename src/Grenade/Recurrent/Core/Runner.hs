{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE RecordWildCards       #-}
module Grenade.Recurrent.Core.Runner (
    trainRecurrent
  , runRecurrent
  ) where

import           Data.Singletons.Prelude
import           Grenade.Core.Network
import           Grenade.Core.Shape

import           Grenade.Recurrent.Core.Network

-- | Drive and network and collect its back propogated gradients.
--
-- TODO: split this nicely into backpropagate and update.
--
-- QUESTION: Should we return a list of gradients or the sum of
--           the gradients? It's different taking into account
--           momentum and L2.
trainRecurrent :: forall shapes layers. SingI (Last shapes)
               => LearningParameters
               -> RecurrentNetwork layers shapes
               -> RecurrentInputs layers
               -> [(S (Head shapes), Maybe (S (Last shapes)))]
               -> (RecurrentNetwork layers shapes, RecurrentInputs layers)
trainRecurrent rate network recinputs examples =
    updateBack $ go inputs network recinputs
  where
    inputs  = fst <$> examples
    targets = snd <$> examples
    updateBack (a,recgrad,_) = (a,updateRecInputs rate recinputs recgrad)

    go  :: forall js sublayers. (Last js ~ Last shapes)
        => [S (Head js)]                 -- ^ input vector
        -> RecurrentNetwork sublayers js -- ^ network to train
        -> RecurrentInputs sublayers
        -> (RecurrentNetwork sublayers js, RecurrentInputs sublayers, [S (Head js)])

    -- This is a simple non-recurrent layer, just map it forwards
    -- Note we're doing training here, we could just return a list of gradients
    -- (and probably will in future).
    go !xs (layer :~~> n) (() :~~+> nIn)
        = let ys                 = runForwards layer <$> xs
              -- recursively run the rest of the network, and get the gradients from above.
              (newFN, ig, grads) = go ys n nIn
              -- calculate the gradient for this layer to pass down,
              back               = uncurry (runBackwards layer) <$> zip (reverse xs) grads
              -- the new trained layer.
              newlayer           = runUpdates rate layer (fst <$> back)

          in (newlayer :~~> newFN, () :~~+> ig, snd <$> back)

    -- This is a recurrent layer, so we need to do a scan, first input to last, providing
    -- the recurrent shape output to the next layer.
    go !xs (layer :~@> n) (g :~@+> nIn)
        = let ys                 = scanlFrom layer g xs

              (newFN, ig, grads) = go (snd <$> ys) n nIn

              backExamples       = zip3 (fst <$> reverse ys) (reverse xs) grads

              (rg, back)         = myscanbackward layer backExamples
              -- the new trained layer.
              newlayer           = runUpdates rate layer (fst <$> back)
          in (newlayer :~@> newFN, rg :~@+> ig, snd <$> back)

    -- Handle the output layer, bouncing the derivatives back down.
    -- We may not have a target for each example, so when we don't use 0 gradient.
    go !xs (OR layer) (ORS ())
        = let ys              = runForwards layer <$> xs
              -- recursively run the rest of the network, and get the gradients from above.
              back            = uncurry (runBackwards layer) <$> zip xs (zipWith makeError ys targets)
              -- the new trained layer.
              newlayer        = runUpdates rate layer (reverse $ fst <$> back)
          in (OR newlayer, ORS (), reverse (snd <$> back))

    go _ _ _ =
      error "Impossible for network and recurrent inputs to have different shapes"


    makeError :: S (Last shapes) -> Maybe (S (Last shapes)) -> S (Last shapes)
    makeError _ Nothing = 0
    makeError y (Just t) = y - t

    updateRecInputs :: forall sublayers.
           LearningParameters
        -> RecurrentInputs sublayers
        -> RecurrentInputs sublayers
        -> RecurrentInputs sublayers

    updateRecInputs l@LearningParameters {..} (() :~~+> xs) (() :~~+> ys)
      = () :~~+> updateRecInputs l xs ys

    updateRecInputs l@LearningParameters {..} (x :~@+> xs) (y :~@+> ys)
      = (realToFrac (learningRate * learningRegulariser) * x - realToFrac learningRate * y) :~@+> updateRecInputs l xs ys

    updateRecInputs _ (ORS ()) (ORS ())
      = ORS ()
    updateRecInputs _ _ _
      = error "Impossible for updateRecInputs to have different shapes"

scanlFrom :: forall x i o. RecurrentLayer x i o
          => x                                  -- ^ the layer
          -> S (RecurrentShape x)               -- ^ place to start
          -> [S i]                              -- ^ list of inputs to scan through
          -> [(S (RecurrentShape x), S o)]      -- ^ list of scan inputs and outputs
scanlFrom !layer !recShape (x:xs) =
  let (lerec, lepush) = runRecurrentForwards layer recShape x
  in  (recShape, lepush) : scanlFrom layer lerec xs
scanlFrom _ _ []      = []

myscanbackward :: forall x i o. RecurrentLayer x i o
                => x                                           -- ^ the layer
                -> [(S (RecurrentShape x), S i, S o)]          -- ^ the list of inputs and output to scan over
                -> (S (RecurrentShape x), [(Gradient x, S i)]) -- ^ list of gradients to fold and inputs to backprop
myscanbackward layer =
  goX 0
    where
  goX :: S (RecurrentShape x) -> [(S (RecurrentShape x), S i, S o)] -> (S (RecurrentShape x), [(Gradient x, S i)])
  goX !lastback ((recShape, lastin, backgrad):xs) =
    let (layergrad, recgrad, ingrad) = runRecurrentBackwards layer recShape lastin lastback backgrad
        (pushedback, ll)             = goX recgrad xs
    in  (pushedback, (layergrad, ingrad) : ll)
  goX !lastback []      = (lastback, [])

-- | Just forwards propagation with no training.
runRecurrent :: RecurrentNetwork layers shapes
             -> RecurrentInputs layers -> S (Head shapes)
             -> (RecurrentInputs layers, S (Last shapes))
runRecurrent (layer :~~> n) (()    :~~+> nr) !x
  = let ys = runForwards layer x
        (nr', o) = runRecurrent n nr ys
    in  (() :~~+> nr', o)
runRecurrent (layer :~@> n) (recin :~@+> nr) !x
  = let (recin', y) = runRecurrentForwards layer recin x
        (nr', o)    = runRecurrent n nr y
    in  (recin' :~@+> nr', o)
runRecurrent (OR layer) (ORS  ()) !x
  = (ORS (), runForwards layer x)

runRecurrent _ _ _
  = error "Impossible for the gradients of a network to have a different length or shape to the network"
