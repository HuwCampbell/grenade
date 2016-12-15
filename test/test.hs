import           Disorder.Core.Main

import qualified Test.Grenade.Layers.Pooling        as Test.Grenade.Layers.Pooling
import qualified Test.Grenade.Layers.Convolution    as Test.Grenade.Layers.Convolution
import qualified Test.Grenade.Layers.FullyConnected as Test.Grenade.Layers.FullyConnected

import qualified Test.Grenade.Layers.Internal.Convolution    as Test.Grenade.Layers.Internal.Convolution
import qualified Test.Grenade.Layers.Internal.Pooling        as Test.Grenade.Layers.Internal.Pooling


main :: IO ()
main =
  disorderMain [
      Test.Grenade.Layers.Pooling.tests
    , Test.Grenade.Layers.Convolution.tests
    , Test.Grenade.Layers.FullyConnected.tests

    , Test.Grenade.Layers.Internal.Convolution.tests
    , Test.Grenade.Layers.Internal.Pooling.tests
    ]
