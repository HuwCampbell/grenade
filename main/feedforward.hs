{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE TupleSections         #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE FlexibleContexts      #-}

import           Control.Monad
import           Control.Monad.Random

import           GHC.TypeLits

import qualified Numeric.LinearAlgebra.Static as SA

import           Options.Applicative

import           Grenade


-- The defininition for our simple feed forward network.
-- The type level list represents the shapes passed through the layers. One can see that for this demonstration
-- we are using relu, tanh and logit non-linear units, which can be easily subsituted for each other in and out.

-- It's important to keep the type signatures, as there's many layers which can "squeeze" into the gaps
-- between the shapes, so inference can't do it all for us.

-- With around 100000 examples, this should show two clear circles which have been learned by the network.
randomNet :: MonadRandom m => m (Network '[ 'D1 2, 'D1 40, 'D1 40, 'D1 10, 'D1 10, 'D1 1, 'D1 1])
randomNet = do
  a :: FullyConnected 2 40  <- randomFullyConnected
  b :: FullyConnected 40 10 <- randomFullyConnected
  c :: FullyConnected 10 1  <- randomFullyConnected
  return $ a :~> Tanh :~> b :~> Relu :~> c :~> O Logit

netTest :: MonadRandom m => LearningParameters -> Int -> m String
netTest rate n = do
    inps <- replicateM n $ do
      s  <- getRandom
      return $ S1D' $ SA.randomVector s SA.Uniform * 2 - 1
    let outs = flip map inps $ \(S1D' v) ->
                 if v `inCircle` (fromRational 0.33, 0.33)  || v `inCircle` (fromRational (-0.33), 0.33)
                   then S1D' $ fromRational 1
                   else S1D' $ fromRational 0
    net0 <- randomNet

    let trained = foldl trainEach net0 (zip inps outs)
    let testIns = [ [ (x,y)  | x <- [0..50] ]
                              | y <- [0..20] ]

    let outMat  = fmap (fmap (\(x,y) -> (render . normx) $ runNet trained (S1D' $ SA.vector [x / 25 - 1,y / 10 - 1]))) testIns
    return $ unlines outMat

  where
    inCircle :: KnownNat n => SA.R n -> (SA.R n, Double) -> Bool
    v `inCircle` (o, r) = SA.norm_2 (v - o) <= r
    trainEach !nt !(i, o) = train rate i o nt

    render n'  | n' <= 0.2  = ' '
               | n' <= 0.4  = '.'
               | n' <= 0.6  = '-'
               | n' <= 0.8  = '='
               | otherwise = '#'

    normx :: S' ('D1 1) -> Double
    normx (S1D' r) = SA.mean r


data FeedForwardOpts = FeedForwardOpts Int LearningParameters

feedForward' :: Parser FeedForwardOpts
feedForward' =
  FeedForwardOpts <$> option auto (long "examples" <> short 'e' <> value 100000)
                  <*> (LearningParameters
                      <$> option auto (long "train_rate" <> short 'r' <> value 0.01)
                      <*> option auto (long "momentum" <> value 0.9)
                      <*> option auto (long "l2" <> value 0.0005)
                      )

main :: IO ()
main = do
    FeedForwardOpts examples rate <- execParser (info (feedForward' <**> helper) idm)
    putStrLn "Training network..."
    putStrLn =<< evalRandIO (netTest rate examples)
