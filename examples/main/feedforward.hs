{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE CPP                   #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE TupleSections         #-}
{-# LANGUAGE TypeFamilies          #-}
import           Control.Monad
import           Control.Monad.Random
import           Data.List ( foldl' )

import qualified Data.ByteString as B
import           Data.Serialize
#if ! MIN_VERSION_base(4,13,0)
import           Data.Semigroup ( (<>) )
#endif
import           GHC.TypeLits

import qualified Numeric.LinearAlgebra.Static as SA

import           Options.Applicative

import           Grenade


-- The defininition for our simple feed forward network.
-- The type level lists represents the layers and the shapes passed through the layers.
-- One can see that for this demonstration we are using relu, tanh and logit non-linear
-- units, which can be easily subsituted for each other in and out.
--
-- With around 100000 examples, this should show two clear circles which have been learned by the network.
type FFNet = Network '[ FullyConnected 2 40, Tanh, FullyConnected 40 10, Relu, FullyConnected 10 1, Logit ]
                     '[ 'D1 2, 'D1 40, 'D1 40, 'D1 10, 'D1 10, 'D1 1, 'D1 1]

randomNet :: MonadRandom m => m FFNet
randomNet = randomNetwork

netTrain :: FFNet -> LearningParameters -> Int -> IO FFNet
netTrain net0 rate n = do
    inps <- replicateM n $ do
      s  <- getRandom
      return $ S1D $ SA.randomVector s SA.Uniform * 2 - 1
    let outs = flip map inps $ \(S1D v) ->
                 if v `inCircle` (fromRational 0.33, 0.33)  || v `inCircle` (fromRational (-0.33), 0.33)
                   then S1D $ fromRational 1
                   else S1D $ fromRational 0

    let trained = foldl' trainEach net0 (zip inps outs)
    return trained

  where
    inCircle :: KnownNat n => SA.R n -> (SA.R n, Double) -> Bool
    v `inCircle` (o, r) = SA.norm_2 (v - o) <= r
    trainEach !network (i,o) = train rate network i o

netLoad :: FilePath -> IO FFNet
netLoad modelPath = do
  modelData <- B.readFile modelPath
  either fail return $ runGet (get :: Get FFNet) modelData

netScore :: FFNet -> IO ()
netScore network = do
    let testIns = [ [ (x,y)  | x <- [0..50] ]
                             | y <- [0..20] ]
        outMat  = fmap (fmap (\(x,y) -> (render . normx) $ runNet network (S1D $ SA.vector [x / 25 - 1,y / 10 - 1]))) testIns
    putStrLn $ unlines outMat

  where
    render n'  | n' <= 0.2  = ' '
               | n' <= 0.4  = '.'
               | n' <= 0.6  = '-'
               | n' <= 0.8  = '='
               | otherwise = '#'

    normx :: S ('D1 1) -> Double
    normx (S1D r) = SA.mean r

data FeedForwardOpts = FeedForwardOpts Int LearningParameters (Maybe FilePath) (Maybe FilePath)

feedForward' :: Parser FeedForwardOpts
feedForward' =
  FeedForwardOpts <$> option auto (long "examples" <> short 'e' <> value 100000)
                  <*> (LearningParameters
                      <$> option auto (long "train_rate" <> short 'r' <> value 0.01)
                      <*> option auto (long "momentum" <> value 0.9)
                      <*> option auto (long "l2" <> value 0.0005)
                      )
                  <*> optional (strOption (long "load"))
                  <*> optional (strOption (long "save"))

main :: IO ()
main = do
    FeedForwardOpts examples rate load save <- execParser (info (feedForward' <**> helper) idm)
    net0 <- case load of
      Just loadFile -> netLoad loadFile
      Nothing -> randomNet

    net <- netTrain net0 rate examples
    netScore net

    case save of
      Just saveFile -> B.writeFile saveFile $ runPut (put net)
      Nothing -> return ()
