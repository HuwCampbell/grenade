{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE ScopedTypeVariables   #-}
import Criterion.Main

import           Grenade
import           Grenade.Recurrent

main :: IO ()
main = do
  input40  :: S ('D1 40)  <- randomOfShape
  lstm     :: RecNet      <- randomRecurrent

  defaultMain [
      bgroup "train"  [ bench "one-time-step"     $ whnf (nfT2 . trainRecurrent lp lstm 0) [(input40, Just input40)]
                      , bench "ten-time-steps"    $ whnf (nfT2 . trainRecurrent lp lstm 0) $ replicate 10 (input40, Just input40)
                      , bench "fifty-time-steps"  $ whnf (nfT2 . trainRecurrent lp lstm 0) $ replicate 50 (input40, Just input40)
                      ]
    ]

nfT2 :: (a, b) -> (a, b)
nfT2 (!a, !b) = (a, b)


type R = Recurrent
type RecNet = RecurrentNetwork '[ R (LSTM 40 512), R (LSTM 512 40) ]
                               '[ 'D1 40, 'D1 512, 'D1 40 ]

lp :: LearningParameters
lp = LearningParameters 0.1 0 0
