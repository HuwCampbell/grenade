module Grenade.Layers.Internal.Update.Accelerate (
    descend
  ) where

import qualified Prelude as P
import Data.Array.Accelerate
import Grenade.Core.LearningParameters.Accelerate

descend
  :: Shape sh
  => Acc LearningParameters
  -> Acc (Array sh Double)
  -> Acc (Array sh Double)
  -> Acc (Array sh Double)
  -> Acc (Array sh Double, Array sh Double)
descend params weights gradient lastUpdate =
  let
    (rate, momentum, regulariser) = unlift params
    outMomentum = zipWith (-) (map ((the momentum) *) lastUpdate) (map ((the rate) *) gradient)
    outWeights = zipWith (-) (zipWith (*) weights outMomentum) (map (((the rate) * (the regulariser)) *) weights)
  in
    lift (outWeights, outMomentum)
