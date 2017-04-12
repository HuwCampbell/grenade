import           Control.Monad

import qualified Test.Grenade.Network

import qualified Test.Grenade.Layers.Pooling
import qualified Test.Grenade.Layers.Convolution
import qualified Test.Grenade.Layers.FullyConnected
import qualified Test.Grenade.Layers.Nonlinear
import qualified Test.Grenade.Layers.PadCrop

import qualified Test.Grenade.Layers.Internal.Convolution
import qualified Test.Grenade.Layers.Internal.Pooling

import qualified Test.Grenade.Recurrent.Layers.LSTM

import           System.Exit
import           System.IO

main :: IO ()
main =
  disorderMain [
      Test.Grenade.Network.tests

    , Test.Grenade.Layers.Pooling.tests
    , Test.Grenade.Layers.Convolution.tests
    , Test.Grenade.Layers.FullyConnected.tests
    , Test.Grenade.Layers.Nonlinear.tests
    , Test.Grenade.Layers.PadCrop.tests

    , Test.Grenade.Layers.Internal.Convolution.tests
    , Test.Grenade.Layers.Internal.Pooling.tests

    , Test.Grenade.Recurrent.Layers.LSTM.tests

    ]

disorderMain :: [IO Bool] -> IO ()
disorderMain tests = do
  lineBuffer
  rs <- sequence tests
  unless (and rs) exitFailure


lineBuffer :: IO ()
lineBuffer = do
  hSetBuffering stdout LineBuffering
  hSetBuffering stderr LineBuffering
