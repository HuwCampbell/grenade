<<<<<<< HEAD
{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE CPP                   #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE TupleSections         #-}
{-# LANGUAGE TypeFamilies          #-}
=======
{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TupleSections       #-}
{-# LANGUAGE TypeFamilies        #-}
{-# LANGUAGE TypeOperators       #-}
>>>>>>> started BatchNorm layer
import           Control.Monad
import           Control.Monad.Random
import           Data.List                    (foldl')

import qualified Data.ByteString              as B
import           Data.Semigroup               ((<>))
import           Data.Serialize
<<<<<<< HEAD
#if ! MIN_VERSION_base(4,13,0)
import           Data.Semigroup ( (<>) )
#endif
=======

>>>>>>> started BatchNorm layer
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
type FFNet = Network '[ FullyConnected 2 40, Relu, FullyConnected 40 10, Relu, FullyConnected 10 1, Logit ]
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
        outMat  = fmap (fmap (\(x,y) -> (render (x/25-1) (y/10-1) . normx) $ runNet network (S1D $ SA.vector [x / 25 - 1,y / 10 - 1]))) testIns
    putStrLn $ unlines outMat

  where
    render x y n'  | x == 0 && y == 0 = '+'
                   | y == 0 = '-'
                   | x == 0 = '|'
                   | n' <= 0.2  = ' '
                   | n' <= 0.4  = '.'
                   | n' <= 0.6  = '-'
                   | n' <= 0.8  = '='
                   | otherwise = '#'

normx :: S ('D1 1) -> Double
normx (S1D r) = SA.mean r

testValues :: FFNet -> IO ()
testValues network = do
  inps <- replicateM 1000 $ do
      s  <- getRandom
      return $ S1D $ SA.randomVector s SA.Uniform * 2 - 1
  let outs = flip map inps $ \(S1D v) ->
                 if v `inCircle` (fromRational 0.33, 0.33)  || v `inCircle` (fromRational (-0.33), 0.33)
                   then 1 :: Integer
                   else 0
  let ress = zip outs (map (round . normx . runNet network) inps)
      correct = length $ filter id $ map (uncurry (==)) ress
      incorrect = length $ filter id $ map (uncurry (/=)) ress
      falsePositives = length $ filter id $ map (uncurry (\shd nn -> shd == 0 && nn == 1)) ress
      falseNegatives = length $ filter id $ map (uncurry (\shd nn -> shd == 1 && nn == 0)) ress
      len = length inps
  -- putStrLn $ "Correct: " ++ show correct ++ "/" ++ show len ++ " = " ++ show (fromIntegral correct / fromIntegral len) ++ "%"
  -- putStrLn $ "Incorrect: " ++ show incorrect ++ "/" ++ show len ++ " = " ++ show (fromIntegral incorrect / fromIntegral len) ++ "%"
  -- putStrLn $ "FalsePositives: " ++ show falsePositives ++ "/" ++ show len ++ " = " ++ show (fromIntegral falsePositives / fromIntegral len) ++ "%"
  -- putStrLn $ "FalseNegatives: " ++ show falseNegatives ++ "/" ++ show len ++ " = " ++ show (fromIntegral falseNegatives / fromIntegral len) ++ "%"

  putStr $ show correct ++ "/" ++ show len ++ " = " ++ show (fromIntegral correct / fromIntegral len) ++ "% | "
  putStr $ show incorrect ++ "/" ++ show len ++ " = " ++ show (fromIntegral incorrect / fromIntegral len) ++ "% | "
  putStr $ show falsePositives ++ "/" ++ show len ++ " = " ++ show (fromIntegral falsePositives / fromIntegral len) ++ "% | "
  putStrLn $ show falseNegatives ++ "/" ++ show len ++ " = " ++ show (fromIntegral falseNegatives / fromIntegral len) ++ "% | "


  where
    inCircle :: KnownNat n => SA.R n -> (SA.R n, Double) -> Bool
    v `inCircle` (o, r) = SA.norm_2 (v - o) <= r


data FeedForwardOpts = FeedForwardOpts Int LearningParameters (Maybe FilePath) (Maybe FilePath)

feedForward' :: Parser FeedForwardOpts
feedForward' =
  FeedForwardOpts <$> option auto (long "examples" <> short 'e' <> value 10000)
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
  -- net0 <- case load of
  --   Just loadFile -> netLoad loadFile
  --   Nothing       -> randomNet

  putStrLn $ "| Nr | Correct | Incorrect | FalseNegatives | FalseNegatives |"
  putStrLn $ "--------------------------------------------------------------"
  mapM_ (\n -> do
    putStr $ "| " ++ show n  ++ " | "
    net0 <- randomNet
    net <- netTrain net0 rate examples
    -- netScore net
    testValues net) [1..100]

  -- case save of
  --   Just saveFile -> B.writeFile saveFile $ runPut (put net)
  --   Nothing       -> return ()

