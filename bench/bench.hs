import Criterion.Main

import           Grenade.Layers.Internal.Convolution
import           Grenade.Layers.Internal.Pooling

import           Numeric.LinearAlgebra

main :: IO ()
main = defaultMain [
    bgroup "im2col" [ bench "im2col 3x4"      $ whnf (im2colUnsafe 2 2 1 1)   ((3><4) [1..])
                    , bench "im2col 28x28"    $ whnf (im2colUnsafe 5 5 1 1)   ((28><28) [1..])
                    , bench "im2col 100x100"  $ whnf (im2colUnsafe 10 10 1 1) ((100><100) [1..])
                    ]
  , bgroup "im2col_c" [ bench "im2col_c 3x4"     $ whnf (im2col 2 2 1 1)   ((3><4) [1..])
                      , bench "im2col_c 28x28"   $ whnf (im2col 5 5 1 1)   ((28><28) [1..])
                      , bench "im2col_c 100x100" $ whnf (im2col 10 10 1 1) ((100><100) [1..])
                      ]
  , bgroup "col2im" [ bench "col2im 3x4"      $ whnf (col2imUnsafe 2 2 1 1 3 4)       ((6><4) [1..])
                    , bench "col2im 28x28"    $ whnf (col2imUnsafe 5 5 1 1 28 28)     ((576><25) [1..])
                    , bench "col2im 100x100"  $ whnf (col2imUnsafe 10 10 1 1 100 100) ((8281><100) [1..])
                    ]
  , bgroup "col2im_c" [ bench "col2im_c 3x4"      $ whnf (col2im 2 2 1 1 3 4)       ((6><4) [1..])
                      , bench "col2im_c 28x28"    $ whnf (col2im 5 5 1 1 28 28)     ((576><25) [1..])
                      , bench "col2im_c 100x100"  $ whnf (col2im 10 10 1 1 100 100) ((8281><100) [1..])
                      ]
  , bgroup "poolfw" [ bench "poolforwards 3x4"      $ whnf (poolForward 2 2 1 1 2 3)     ((3><4) [1..])
                    , bench "poolforwards 28x28"    $ whnf (poolForward 5 5 1 1 4 24)    ((28><28) [1..])
                    , bench "poolforwards 100x100"  $ whnf (poolForward 10 10 1 1 91 91) ((100><100) [1..])
                    ]
  , bgroup "poolbw" [ bench "poolbackwards 3x4"      $ whnf (poolBackward 2 2 1 1   ((3><4) [1..]))     ((2><3) [1..])
                    , bench "poolbackwards 28x28"    $ whnf (poolBackward 5 5 1 1   ((28><28) [1..]))   ((24><24) [1..])
                    , bench "poolbackwards 100x100"  $ whnf (poolBackward 10 10 1 1 ((100><100) [1..])) ((91><91) [1..])
                    ]
  ]
