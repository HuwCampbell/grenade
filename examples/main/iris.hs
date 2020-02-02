{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE CPP                   #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE DeriveGeneric         #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE TupleSections         #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE OverloadedStrings     #-}

module Main where

import           Control.Applicative
import           Control.Monad
import           Control.Monad.Random

import           Data.List ( foldl' )
import           Data.Maybe ( mapMaybe )
#if ! MIN_VERSION_base(4,13,0)
import           Data.Semigroup ( (<>) )
#endif
import           Data.Csv ( FromField, FromRecord, decode, parseField, HasHeader(..) )
import           Data.ByteString.Lazy.Char8 (pack)
import qualified Data.Vector as V
import qualified Data.Vector.Storable as SV

import           Numeric.LinearAlgebra ( maxIndex )
import qualified Numeric.LinearAlgebra.Static as SA

import           Options.Applicative
import           System.FilePath ( (</>) )
import           System.Random.Shuffle (shuffleM)
import           GHC.Generics (Generic)
import           GHC.Float ( float2Double)

import           Grenade
import           Grenade.Utils.OneHot


-- Adapted from https://mmhaskell.com/machine-learning/deep-learning

-- It's logistic regression!
--
-- This network is used to show how we can embed a Network as a layer in the larger IrisNetwork
-- type.

type IrisNetwork
  = Network
      '[FullyConnected 4 10, Relu, FullyConnected 10 3]
      '[ 'D1 4, 'D1 10, 'D1 10, 'D1 3]

type IrisRow = (S ( 'D1 4), S ( 'D1 3))

randomIris :: MonadRandom m => m IrisNetwork
randomIris = randomNetwork

runIris :: Int -> FilePath -> Int -> LearningParameters -> IO ()
runIris iterations dataDir nSamples rate = do
  records <- readIrisFromFile (dataDir </> "iris.data")
  let numRecords = V.length records
  shuffledRecords <- chooseRandomRecords records numRecords
  let (trainRecords, validateRecords) = V.splitAt nSamples shuffledRecords

  let trainData = mapMaybe parseRecord (V.toList trainRecords)
  let validateData = mapMaybe parseRecord (V.toList validateRecords)

  if length trainData
       /= length trainRecords
       || length validateData
       /= length validateRecords
    then putStrLn
      "Parsing train data or validation data could not be fully parsed"
    else do
      initialNetwork <- randomIris
      foldM_ (run trainData validateData) initialNetwork [1 .. iterations]
 where

  run :: [IrisRow] -> [IrisRow] -> IrisNetwork -> Int -> IO IrisNetwork
  run trainData validateData network iterationNum = do
    sampledData <- V.toList
      <$> chooseRandomRecords (V.fromList trainData) (nSamples * 3 `div` 4)
    -- Slower drop the learning rate
    let rate' = rate { learningRate = learningRate rate * 0.99 ^ iterationNum }
    let newNetwork     = foldl' (trainRow rate') network sampledData
    let labelVectors   = fmap (testRow newNetwork) validateData
    let labelValues    = fmap getLabels labelVectors
    let total          = length labelValues
    let correctEntries = length $ filter ((==) <$> fst <*> snd) labelValues
    putStrLn $ "Iteration: " ++ show iterationNum
    putStrLn $ show correctEntries ++ " correct out of: " ++ show total
    return newNetwork

  trainRow :: LearningParameters -> IrisNetwork -> IrisRow -> IrisNetwork
  trainRow lp network (input, output) = train lp network input output

  -- Takes a test row, returns predicted output and actual output from the network.
  testRow :: IrisNetwork -> IrisRow -> (S ( 'D1 3), S ( 'D1 3))
  testRow net (rowInput, predictedOutput) =
    (predictedOutput, runNet net rowInput)

  -- Goes from probability output vector to label
  getLabels :: (S ( 'D1 3), S ( 'D1 3)) -> (Int, Int)
  getLabels (S1D predictedLabel, S1D actualOutput) =
    (maxIndex (SA.extract predictedLabel), maxIndex (SA.extract actualOutput))



data IrisOpts = IrisOpts FilePath Int Int LearningParameters

iris' :: Parser IrisOpts
iris' =
  IrisOpts
    <$> argument str (metavar "DATADIR")
        -- How many samples from the dataset should be used for training?
        -- (The rest are used for validation)
    <*> option auto (long "training_samples" <> short 't' <> value 100)
    <*> option auto (long "iterations" <> short 'i' <> value 15)
    <*> (   LearningParameters
        <$> option auto (long "train_rate" <> short 'r' <> value 0.01)
        <*> option auto (long "momentum" <> value 0.9)
        <*> option auto (long "l2" <> value 0.0005)
        )

main :: IO ()
main = do
  IrisOpts dataDir nSamples iter rate <- execParser
    (info (iris' <**> helper) idm)
  putStr "Training convolutional neural network with "
  putStr $ show nSamples
  putStrLn " samples..."

  runIris iter dataDir nSamples rate

data IrisClass = Setosa | Versicolor | Virginica
  deriving (Show, Read, Eq, Ord, Generic, Enum, Bounded)

data IrisRecord = IrisRecord {
    sepalLength :: Float,
    sepalWidth  :: Float,
    petalLength :: Float,
    petalWidth  :: Float,
    specie      :: IrisClass
} deriving (Generic, Show, Read)

instance FromRecord IrisRecord

instance FromField IrisClass where
  parseField "Iris-setosa"     = return Setosa
  parseField "Iris-versicolor" = return Versicolor
  parseField "Iris-virginica"  = return Virginica
  parseField _                 = fail "unknown iris class"

parseRecord :: IrisRecord -> Maybe IrisRow
parseRecord record = case (input, output) of
  (Just i, Just o) -> Just (i, o)
  _                -> Nothing
 where
  input =
    fromStorable
      $   SV.fromList
      $   float2Double
      <$> [ sepalLength record / 8.0
          , sepalWidth record / 8.0
          , petalLength record / 8.0
          , petalWidth record / 8.0
          ]
  output = oneHot (fromEnum $ specie record)

readIrisFromFile :: FilePath -> IO (V.Vector IrisRecord)
readIrisFromFile fp = do
  contents <- readFile fp
  let contentsAsBs = pack contents
  let results =
        decode HasHeader contentsAsBs :: Either String (V.Vector IrisRecord)
  case results of
    Left  err     -> error err
    Right records -> return records


-- A function that takes this vector of records, and selects sampleSize of them at random.
chooseRandomRecords :: V.Vector a -> Int -> IO (V.Vector a)
chooseRandomRecords records sampleSize = do
  let numRecords = V.length records
  chosenIndices <- take sampleSize <$> shuffleM [0 .. (numRecords - 1)]
  return . V.fromList $ (records V.!) <$> chosenIndices

