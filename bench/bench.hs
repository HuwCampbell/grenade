{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE ScopedTypeVariables #-}
import           Control.Monad
import           Control.Monad.Random
import           Criterion.Main
import           Data.Default
import           Data.List                           (foldl')
import           Data.Singletons
import           Data.Singletons.Prelude.List
import           GHC.TypeLits
import qualified Numeric.LinearAlgebra.Static        as SA


import           Grenade


import           Grenade.Layers.Internal.Convolution
import           Grenade.Layers.Internal.Pooling

import           Numeric.LinearAlgebra

ff :: SpecNet
ff = specFullyConnected 2 40 |=> specTanh1D 40 |=> netSpecInner |=> specFullyConnected 20 30 |=> specRelu1D 30 |=> specFullyConnected 30 20 |=> specRelu1D 20 |=> specFullyConnected 20 10 |=> specRelu1D 10 |=> specFullyConnected 10 1 |=> specLogit1D 1 |=> specNil1D 1
  where netSpecInner = specFullyConnected 40 30 |=> specRelu1D 30 |=> specFullyConnected 30 20 |=> specNil1D 20

hugeFf :: SpecNet
hugeFf = specFullyConnected 2 150 |=> specTanh1D 150 |=> specFullyConnected 150 150 |=> specRelu1D 150 |=> specFullyConnected 150 150 |=> specRelu1D 150 |=> specFullyConnected 150 150 |=> specRelu1D 150 |=> specFullyConnected 150 150 |=> specRelu1D 150 |=> specFullyConnected 150 150 |=> specRelu1D 150 |=> specFullyConnected 150 150 |=> specRelu1D 150 |=> specFullyConnected 150 150 |=> specRelu1D 150 |=> specFullyConnected 150 150 |=> specRelu1D 150 |=> specFullyConnected 150 150 |=> specRelu1D 150 |=> specFullyConnected 150 150 |=> specRelu1D 150 |=> specFullyConnected 150 30 |=> specRelu1D 30 |=> specFullyConnected 30 20 |=> specRelu1D 20 |=> specFullyConnected 20 10 |=> specRelu1D 10 |=> specFullyConnected 10 1 |=> specLogit1D 1 |=> specNil1D 1

hugeBigFf :: SpecNet
hugeBigFf = specFullyConnected 2 1500 |=> specTanh1D 1500 |=> specFullyConnected 1500 750 |=> specRelu1D 750 |=> specFullyConnected 750 150 |=> specRelu1D 150 |=> specFullyConnected 150 150 |=> specRelu1D 150 |=> specFullyConnected 150 150 |=> specRelu1D 150 |=> specFullyConnected 150 150 |=> specRelu1D 150 |=> specFullyConnected 150 150 |=> specRelu1D 150 |=> specFullyConnected 150 150 |=> specRelu1D 150 |=> specFullyConnected 150 150 |=> specRelu1D 150 |=> specFullyConnected 150 150 |=> specRelu1D 150 |=> specFullyConnected 150 150 |=> specRelu1D 150 |=> specFullyConnected 150 30 |=> specRelu1D 30 |=> specFullyConnected 30 20 |=> specRelu1D 20 |=> specFullyConnected 20 10 |=> specRelu1D 10 |=> specFullyConnected 10 1 |=> specLogit1D 1 |=> specNil1D 1


main :: IO ()
main = do
  putStrLn $ "Benchmarking with type: " ++ nameF
  x :: S ('D2 60 60) <- randomOfShape
  y :: S ('D3 60 60 1) <- randomOfShape
  SpecConcreteNetwork1D1D netFF <- networkFromSpecificationWith def ff
  SpecConcreteNetwork1D1D netHuge <- networkFromSpecificationWith def hugeFf
  SpecConcreteNetwork1D1D netHugeBig <- networkFromSpecificationWith def hugeBigFf
  SpecConcreteNetwork1D1D netHugeHBlas <- networkFromSpecificationWith (def { cpuBackend = HBLAS }) hugeFf
  SpecConcreteNetwork1D1D netHugeBigHBlas <- networkFromSpecificationWith (def { cpuBackend = HBLAS }) hugeBigFf
  defaultMain
    [ -- bgroup
    --     "im2col"
    --     [ bench "im2col 3x4" $ whnf (im2col 2 2 1 1) ((3 >< 4) [1 ..])
    --     , bench "im2col 28x28" $ whnf (im2col 5 5 1 1) ((28 >< 28) [1 ..])
    --     , bench "im2col 100x100" $ whnf (im2col 10 10 1 1) ((100 >< 100) [1 ..])
    --     ]
    -- , bgroup
    --     "col2im"
    --     [ bench "col2im 3x4" $ whnf (col2im 2 2 1 1 3 4) ((6 >< 4) [1 ..])
    --     , bench "col2im 28x28" $ whnf (col2im 5 5 1 1 28 28) ((576 >< 25) [1 ..])
    --     , bench "col2im 100x100" $ whnf (col2im 10 10 1 1 100 100) ((8281 >< 100) [1 ..])
    --     ]
    -- , bgroup
    --     "poolfw"
    --     [ bench "poolforwards 3x4" $ whnf (poolForward 1 3 4 2 2 1 1) ((3 >< 4) [1 ..])
    --     , bench "poolforwards 28x28" $ whnf (poolForward 1 28 28 5 5 1 1) ((28 >< 28) [1 ..])
    --     , bench "poolforwards 100x100" $ whnf (poolForward 1 100 100 10 10 1 1) ((100 >< 100) [1 ..])
    --     ]
    -- , bgroup
    --     "poolbw"
    --     [ bench "poolbackwards 3x4" $ whnf (poolBackward 1 3 4 2 2 1 1 ((3 >< 4) [1 ..])) ((2 >< 3) [1 ..])
    --     , bench "poolbackwards 28x28" $ whnf (poolBackward 1 28 28 5 5 1 1 ((28 >< 28) [1 ..])) ((24 >< 24) [1 ..])
    --     , bench "poolbackwards 100x100" $ whnf (poolBackward 1 100 100 10 10 1 1 ((100 >< 100) [1 ..])) ((91 >< 91) [1 ..])
    --     ]
    -- , bgroup
    --     "padcrop"
    --     [ bench "pad 2D 60x60" $ whnf (testRun2D Pad) x
    --     , bench "pad 3D 60x60" $ whnf (testRun3D Pad) y
    --     , bench "crop 2D 60x60" $ whnf (testRun2D' Crop) x
    --     , bench "crop 3D 60x60" $ whnf (testRun3D' Crop) y
    --     ]
    -- , bgroup
    --     "feedforward SGD"
    --     [ bench "ANN 1000 training steps" $ nfIO $ netTrain netFF defSGD 1000
    --     , bench "ANN 10000 training steps" $ nfIO $ netTrain netFF defSGD 10000
    --     , bench "ANN Huge 100 train steps" $ nfIO $ netTrain netHuge defSGD 100
    --     , bench "ANN Huge 1000 train steps" $ nfIO $ netTrain netHuge defSGD 1000
    --     ]
    -- ,
      bgroup
        "feedforward Adam"
        [ bench "ANN 100 Huge train steps" $ nfIO $ netTrain netHuge defAdam 100
        , bench "ANN 100 Huge HBLAS train steps" $ nfIO $ netTrain netHugeHBlas defAdam 100
        , bench "ANN 100 Huge Big train steps" $ nfIO $ netTrain netHugeBig defAdam 100
        , bench "ANN 100 Huge HBLAS Big train steps" $ nfIO $ netTrain netHugeBigHBlas defAdam 100
        , bench "ANN 1000 Huge train steps" $ nfIO $ netTrain netHuge defAdam 1000
        , bench "ANN 1000 Huge HBLAS train steps" $ nfIO $ netTrain netHugeHBlas defAdam 1000
        , bench "ANN 1000 Huge Big train steps" $ nfIO $ netTrain netHugeBig defAdam 1000
        , bench "ANN 1000 Huge HBLAS Big train steps" $ nfIO $ netTrain netHugeBigHBlas defAdam 1000
        , bench "ANN 1000 training steps" $ nfIO $ netTrain netFF defAdam 1000
        , bench "ANN 10000 training steps" $ nfIO $ netTrain netFF defAdam 10000
        ]
    ]
  putStrLn $ "Benchmarked with type: " ++ nameF

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

netTrain ::
     (SingI (Last shapes), MonadRandom m, KnownNat len1, KnownNat len2, Head shapes ~ 'D1 len1, Last shapes ~ 'D1 len2)
  => Network layers shapes
  -> Optimizer o
  -> Int
  -> m (Network layers shapes)
netTrain net0 op n = do
  inps <-
    replicateM n $ do
      s <- getRandom
      return $ S1D $ SA.randomVector s SA.Uniform * 2 - 1
  let outs =
        flip map inps $ \(S1D v) ->
          if v `inCircle` (fromRational 0.50, 0.50) || v `inCircle` (fromRational (-0.50), 0.50)
            then S1D $ fromRational 1
            else S1D $ fromRational 0
  let trained = foldl' trainEach net0 (zip inps outs)
  return trained
  where
    trainEach !network (i, o) = train op network i o
    inCircle :: KnownNat n => SA.R n -> (SA.R n, RealNum) -> Bool
    v `inCircle` (o, r) = SA.norm_2 (v - o) <= r
