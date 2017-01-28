import           Disorder.Core.Main

import qualified Test.Grenade.Layers.Pooling
import qualified Test.Grenade.Layers.Convolution
import qualified Test.Grenade.Layers.FullyConnected
import qualified Test.Grenade.Layers.Nonlinear

import qualified Test.Grenade.Layers.Internal.Convolution
import qualified Test.Grenade.Layers.Internal.Pooling

import qualified Test.Grenade.Recurrent.Layers.LSTM

main :: IO ()
main =
  disorderMain [
      Test.Grenade.Layers.Pooling.tests
    , Test.Grenade.Layers.Convolution.tests
    , Test.Grenade.Layers.FullyConnected.tests
    , Test.Grenade.Layers.Nonlinear.tests

    , Test.Grenade.Layers.Internal.Convolution.tests
    , Test.Grenade.Layers.Internal.Pooling.tests

    , Test.Grenade.Recurrent.Layers.LSTM.tests
    ]
