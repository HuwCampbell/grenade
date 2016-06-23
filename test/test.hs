import           Disorder.Core.Main

import qualified Test.Grenade.Layers.Pooling     as Test.Grenade.Layers.Pooling
import qualified Test.Grenade.Layers.Convolution as Test.Grenade.Layers.Convolution

main :: IO ()
main =
  disorderMain [
      Test.Grenade.Layers.Pooling.tests
    , Test.Grenade.Layers.Convolution.tests
    ]
