{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE ScopedTypeVariables   #-}
import Criterion.Main

import           Grenade
import           Grenade.Accelerate as GA

import           Grenade.Layers.Internal.Convolution
import qualified Grenade.Layers.Internal.Convolution.Accelerate as CA
import           Grenade.Layers.Internal.Pooling
import qualified Grenade.Layers.Internal.Pooling.Accelerate as PA

import qualified Data.Array.Accelerate as A
import           Data.Array.Accelerate (Z(..), (:.)(..))
import           Data.Array.Accelerate.Interpreter as I
import           Data.Array.Accelerate.LLVM.Native as LN
--import           Data.Array.Accelerate.LLVM.PTX as LP

import           Numeric.LinearAlgebra

main :: IO ()
main = do
  x    :: S ('D2 60 60  )  <- randomOfShape
  y    :: S ('D3 60 60 1)  <- randomOfShape

  defaultMain
    [ bgroup "linear algebra"
        [ bgroup "im2col"
            [ bench "im2col 3x4"     $ nf (im2col 2 2 1 1)   ((3><4) [1..])
            , bench "im2col 28x28"   $ nf (im2col 5 5 1 1)   ((28><28) [1..])
            , bench "im2col 100x100" $ nf (im2col 10 10 1 1) ((100><100) [1..])
            ]
        , bgroup "col2im"
            [ bench "col2im 3x4"      $ nf (col2im 2 2 1 1 3 4)       ((6><4) [1..])
            , bench "col2im 28x28"    $ nf (col2im 5 5 1 1 28 28)     ((576><25) [1..])
            , bench "col2im 100x100"  $ nf (col2im 10 10 1 1 100 100) ((8281><100) [1..])
            ]
        , bgroup "poolfw"
            [ bench "poolforwards 3x4"      $ nf (poolForward 1 3 4 2 2 1 1)     ((3><4) [1..])
            , bench "poolforwards 28x28"    $ nf (poolForward 1 28 28 5 5 1 1)    ((28><28) [1..])
            , bench "poolforwards 100x100"  $ nf (poolForward 1 100 100 10 10 1 1) ((100><100) [1..])
            ]
        , bgroup "poolbw"
            [ bench "poolbackwards 3x4"      $ nf (poolBackward 1 3 4 2 2 1 1   ((3><4) [1..]))     ((2><3) [1..])
            , bench "poolbackwards 28x28"    $ nf (poolBackward 1 28 28 5 5 1 1   ((28><28) [1..]))   ((24><24) [1..])
            , bench "poolbackwards 100x100"  $ nf (poolBackward 1 100 100 10 10 1 1 ((100><100) [1..])) ((91><91) [1..])
            ]
        , bgroup "padcrop"
            [ bench "pad 2D 60x60"    $ nf (testRun2D Pad) x
            , bench "pad 3D 60x60"    $ nf (testRun3D Pad) y
            , bench "crop 2D 60x60"   $ nf (testRun2D' Crop) x
            , bench "crop 3D 60x60"   $ nf (testRun3D' Crop) y
            ]
        ]
    , bgroup "accelerate"
        [ bgroup name
            [ bgroup "im2col"
                [ bench "im2col 3x4"     $ nf (run . CA.im2col 2 2 1 1)   (A.use $ A.fromList (Z :. 3 :. 4) [1..])
                , bench "im2col 28x28"   $ nf (run . CA.im2col 5 5 1 1)   (A.use $ A.fromList (Z :. 28 :. 28) [1..])
                , bench "im2col 100x100" $ nf (run . CA.im2col 10 10 1 1) (A.use $ A.fromList (Z :. 100 :. 100) [1..])
                ]
            ]
        | (name, run) <-
            [ ("interpreter", I.run)
            , ("llvm-native", LN.run)
            --, ("llvm-ptx", LP.run1)
            ]
        ]
    ]


testRun2D :: Pad 1 1 1 1 -> S ('D2 60 60) -> S ('D2 62 62)
testRun2D = snd ... runForwards

testRun3D :: Pad 1 1 1 1 -> S ('D3 60 60 1) -> S ('D3 62 62 1)
testRun3D = snd ... runForwards

testRun2D' :: Crop 1 1 1 1 -> S ('D2 60 60) -> S ('D2 58 58)
testRun2D' = snd ... runForwards

testRun3D' :: Crop 1 1 1 1 -> S ('D3 60 60 1) -> S ('D3 58 58 1)
testRun3D' = snd ... runForwards

(...) :: (a -> b) -> (c -> d -> a) -> c -> d -> b
(...) = (.) . (.)
