{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE ScopedTypeVariables   #-}
import Criterion.Main

import           Grenade
import           Grenade.Recurrent
import           Grenade.Layers.Internal.Update

import qualified Numeric.LinearAlgebra.Static as H

main :: IO ()
main = do
  layer60  :: LSTM 40 60  <- createRandom
  layer512 :: LSTM 40 512 <- createRandom
  input40  :: S ('D1 40)  <- randomOfShape
  rec60    :: S ('D1 60)  <- randomOfShape
  rec512   :: S ('D1 512) <- randomOfShape
  (lstm, lstm')  :: (RecNet, RecInput) <- randomRecurrent

  let upIn60   :: H.R 3600    = H.randomVector 1 H.Uniform * 2 - 1
  let upIn512  :: H.R 262144  = H.randomVector 1 H.Uniform * 2 - 1

  defaultMain [
      bgroup "lstm"   [ bench "forwards-60"       $ nf (nfT2 . uncurry  (testRun60   layer60))  (rec60, input40)
                      , bench "forwards-512"      $ nf (nfT2 . uncurry  (testRun512  layer512)) (rec512, input40)
                      , bench "backwards-60"      $ nf (nfT3 . uncurry4 (testRun60'  layer60))  (rec60, input40, rec60, rec60)
                      , bench "backwards-512"     $ nf (nfT3 . uncurry4 (testRun512' layer512)) (rec512, input40, rec512, rec512)
                      ]
    , bgroup "update" [ bench "matrix-60x60"      $ nf (uncurry3 (decendVector 1 1 1)) (upIn60, upIn60, upIn60)
                      , bench "matrix-512x512"    $ nf (uncurry3 (decendVector 1 1 1)) (upIn512, upIn512, upIn512)
                      ]
    , bgroup "train"  [ bench "one-time-step"     $ whnf (nfT2 . trainRecurrent lp lstm lstm') [(input40, Just input40)]
                      , bench "ten-time-steps"    $ whnf (nfT2 . trainRecurrent lp lstm lstm') $ replicate 10 (input40, Just input40)
                      , bench "fifty-time-steps"  $ whnf (nfT2 . trainRecurrent lp lstm lstm') $ replicate 50 (input40, Just input40)
                      ]
    ]

testRun60 :: LSTM 40 60 -> S ('D1 60) -> S ('D1 40) -> (S ('D1 60), S ('D1 60))
testRun60 = runRecurrentForwards

testRun60' :: LSTM 40 60 -> S ('D1 60) -> S ('D1 40) -> S ('D1 60) -> S ('D1 60) -> (Gradient (LSTM 40 60), S ('D1 60), S ('D1 40))
testRun60' = runRecurrentBackwards

testRun512 :: LSTM 40 512 -> S ('D1 512) -> S ('D1 40) -> (S ('D1 512), S ('D1 512))
testRun512 = runRecurrentForwards

testRun512' :: LSTM 40 512 -> S ('D1 512) -> S ('D1 40) -> S ('D1 512) -> S ('D1 512) -> (Gradient (LSTM 40 512), S ('D1 512), S ('D1 40))
testRun512' = runRecurrentBackwards

uncurry4 :: (t -> t1 -> t2 -> t3 -> t4) -> (t, t1, t2, t3) -> t4
uncurry4 f (a,b,c,d) = f a b c d

uncurry3 :: (t -> t1 -> t2 -> t3) -> (t, t1, t2) -> t3
uncurry3 f (a,b,c) = f a b c

nfT2 :: (a, b) -> (a, b)
nfT2 (!a, !b) = (a, b)

nfT3 :: (a, b, c) -> (b, c)
nfT3 (!_, !b, !c) = (b, c)


type F = FeedForward
type R = Recurrent
type RecNet = RecurrentNetwork '[ R (LSTM 40 512), R (LSTM 512 40), F (FullyConnected 40 40), F Logit]
                               '[ 'D1 40, 'D1 512, 'D1 40, 'D1 40, 'D1 40 ]

type RecInput = RecurrentInputs '[ R (LSTM 40 512), R (LSTM 512 40), F (FullyConnected 40 40), F Logit]

lp :: LearningParameters
lp = LearningParameters 0.1 0 0
