{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TupleSections       #-}
{-# LANGUAGE TypeFamilies        #-}
{-# LANGUAGE TypeOperators       #-}
import           Control.Monad
import           Control.Monad.Random
import           Data.List                    (foldl')

import qualified Data.ByteString              as B
import           Data.Semigroup               ((<>))
import           Data.Serialize

import           GHC.TypeLits

import qualified Numeric.LinearAlgebra.Static as SA

import           Options.Applicative

import           Grenade

-- The definition for our simple feed forward network.
-- The type level lists represents the layers and the shapes passed through the layers.
-- One can see that for this demonstration we are using relu, tanh and logit non-linear
-- units, which can be easily substitute for each other in and out.
--
-- With around 100000 examples, this should show two clear circles which have been learned by the network.
type FFNet = Network '[ FullyConnected 2 40, Tanh, FullyConnected 40 30, Relu, FullyConnected 30 20, Relu, FullyConnected 20 10, Relu, FullyConnected 10 1, Logit ]
                     '[ 'D1 2, 'D1 40, 'D1 40, 'D1 30, 'D1 30, 'D1 20, 'D1 20, 'D1 10, 'D1 10, 'D1 1, 'D1 1]

randomNet :: IO FFNet
randomNet = randomNetworkInitWith HeEtAl  -- you might want to try `Xavier` or `UniformInit` instead of `HeEtAl`


netTrain :: FFNet -> LearningParameters -> Int -> IO FFNet
netTrain net0 rate n = do
    inps <- replicateM n $ do
      s  <- getRandom
      return $ S1D $ SA.randomVector s SA.Uniform * 2 - 1
    let outs = flip map inps $ \(S1D v) ->
                 if v `inCircle` (fromRational 0.50, 0.50)  || v `inCircle` (fromRational (-0.50), 0.50)
                   then S1D $ fromRational 1
                   else S1D $ fromRational 0

    let trained = foldl' trainEach net0 (zip inps outs)
    return trained

  where trainEach !network (i,o) = train rate network i o

netLoad :: FilePath -> IO FFNet
netLoad modelPath = do
  modelData <- B.readFile modelPath
  either fail return $ runGet (get :: Get FFNet) modelData

renderClass :: IO ()
renderClass = do
  let testIns = [ [ (x,y)  | x <- [0..50] ]
                           | y <- [0..20] ]
  let outMat  = fmap (fmap (\(x,y) -> (render (x/25-1) (y/10-1)))) testIns
  putStrLn $ unlines outMat

  where
    render x y  | x == 0 && y == 0 = '+'
                | y == 0 = '-'
                | x == 0 = '|'
                | otherwise = let v = SA.vector [x,y] :: SA.R 2
                              in if v `inCircle` (fromRational 0.50, 0.50)  || v `inCircle` (fromRational (-0.50), 0.50)
                                 then '1'
                                 else ' '
-- netLoad :: FilePath -> IO FFNet
-- netLoad modelPath = do
--   modelData <- B.readFile modelPath
--   either fail return $ runGet (get :: Get FFNet) modelData

-- renderClass :: IO ()
-- renderClass = do
--   let testIns = [ [ (x,y)  | x <- [0..50] ]
--                            | y <- [0..20] ]
--   let outMat  = fmap (fmap (\(x,y) -> (render (x/25-1) (y/10-1)))) testIns
--   putStrLn $ unlines outMat

--   where
--     render x y  | x == 0 && y == 0 = '+'
--                 | y == 0 = '-'
--                 | x == 0 = '|'
--                 | otherwise = let v = SA.vector [x,y] :: SA.R 2
--                               in if v `inCircle` (fromRational 0.50, 0.50)  || v `inCircle` (fromRational (-0.50), 0.50)
--                                  then '1'
--                                  else ' '


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
                 if v `inCircle` (fromRational 0.50, 0.50)  || v `inCircle` (fromRational (-0.50), 0.50)
                   then 1 :: Integer
                   else 0
  let ress = zip outs (map (round . normx . runNet network) inps)
      correct = length $ filter id $ map (uncurry (==)) ress
      incorrect = length $ filter id $ map (uncurry (/=)) ress
      falsePositives = length $ filter id $ map (uncurry (\shd nn -> shd == 0 && nn == 1)) ress
      falseNegatives = length $ filter id $ map (uncurry (\shd nn -> shd == 1 && nn == 0)) ress
  putStr $ show correct  ++ " | "
  putStr $ show incorrect ++ " | "
  putStr $ show falsePositives ++ " | "
  putStrLn $ show falseNegatives ++ " | "


inCircle :: KnownNat n => SA.R n -> (SA.R n, Double) -> Bool
v `inCircle` (o, r) = SA.norm_2 (v - o) <= r


data FeedForwardOpts = FeedForwardOpts Int LearningParameters

feedForward' :: Parser FeedForwardOpts
feedForward' =
  FeedForwardOpts <$> option auto (long "examples" <> short 'e' <> value 1000)
                  <*> (LearningParameters
                       <$> option auto (long "train_rate" <> short 'r' <> value 0.005)
                       <*> option auto (long "momentum" <> value 0.0)
                       <*> option auto (long "l2" <> value 0.0005)
                      )


main :: IO ()
main = do
  FeedForwardOpts examples rate <- execParser (info (feedForward' <**> helper) idm)

  putStrLn "| Nr | Correct | Incorrect | FalsePositives | FalseNegatives |"
  putStrLn "--------------------------------------------------------------"
  let nr = 100 :: Int
  mapM_ (\n -> do
    putStr $ "| " ++ show n  ++ " | "
    net0 <- randomNet
    net <- netTrain net0 rate examples
    -- netScore net
    testValues net) [0..nr-1]


  n2 <- randomNetwork :: IO (Network '[FullyConnected 1 10] '[ 'D1 1, 'D1 10 ])
  let spec = networkToSpecification n2
  print spec

  SpecNetwork n3 <- networkFromSpecification spec
  print $ networkToSpecification n3

  let spec' = SpecNCons (specFullyConnected 10 30) (specNil1D 30)
      spec'' = specFullyConnected 7 30 |=> specElu1D 30 |=> (specFullyConnected 30 50 |=> specRelu1D 50 |=> specFullyConnected 50 30) |=>
               specNil1D 30
  print spec'
  print spec''
  n4 <- networkFromSpecification spec''
  print n4
