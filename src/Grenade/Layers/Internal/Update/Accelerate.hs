module Grenade.Layers.Internal.Update.Accelerate (
    descend
  ) where

import qualified Prelude as P
import Data.Array.Accelerate
import Grenade.Core.LearningParameters

descend
  :: Shape sh
  => AccelLearningParameters
  -> Acc (Array sh Double)
  -> Acc (Array sh Double)
  -> Acc (Array sh Double)
  -> Acc (Array sh Double, Array sh Double)
descend params weights gradient lastUpdate =
  let
    rate = params !! 0
    momentum = params !! 1
    regulariser = params !! 2
    outMomentum = zipWith (-) (map (momentum *) lastUpdate) (map (rate *) gradient)
    outWeights = zipWith (-) (zipWith (*) weights outMomentum) (map ((rate * regulariser) *) weights)
  in
    lift (outWeights, outMomentum)