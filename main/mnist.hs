{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE TupleSections         #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE FlexibleContexts      #-}

import           Control.Applicative
import           Control.Monad
import           Control.Monad.Random
import           Control.Monad.Trans.Class
import           Control.Monad.Trans.Except

import qualified Data.Attoparsec.Text as A
import qualified Data.Text as T
import qualified Data.Text.IO as T

import           Numeric.LinearAlgebra (maxIndex)
import qualified Numeric.LinearAlgebra.Static as SA

import           Options.Applicative

import           Grenade

-- The definition of our convolutional neural network.
-- In the type signature, we have a type level list of shapes which are passed between the layers.
-- One can see that the images we are inputing are two dimensional with 28 * 28 pixels.

-- It's important to keep the type signatures, as there's many layers which can "squeeze" into the gaps
-- between the shapes, so inference can't do it all for us.

-- With the mnist data from Kaggle normalised to doubles between 0 and 1, learning rate of 0.01 and 15 iterations,
-- this network should get down to about a 1.3% error rate.
type MNIST = Network '[ Convolution 1 10 5 5 1 1, Pooling 2 2 2 2, Relu, Convolution 10 16 5 5 1 1, Pooling 2 2 2 2, FlattenLayer, Relu, FullyConnected 256 80, Logit, FullyConnected 80 10, Logit]
                     '[ 'D2 28 28, 'D3 24 24 10, 'D3 12 12 10, 'D3 12 12 10, 'D3 8 8 16, 'D3 4 4 16, 'D1 256, 'D1 256, 'D1 80, 'D1 80, 'D1 10, 'D1 10]

randomMnist :: MonadRandom m => m MNIST
randomMnist = randomNetwork

convTest :: Int -> FilePath -> FilePath -> LearningParameters -> ExceptT String IO ()
convTest iterations trainFile validateFile rate = do
  net0         <- lift randomMnist
  trainData    <- readMNIST trainFile
  validateData <- readMNIST validateFile
  lift $ foldM_ (runIteration trainData validateData) net0 [1..iterations]

    where
  trainEach rate' !network (i, o) = train rate' network i o

  runIteration trainRows validateRows net i = do
    let trained' = foldl (trainEach ( rate { learningRate = learningRate rate * 0.9 ^ i} )) net trainRows
    let res      = fmap (\(rowP,rowL) -> (rowL,) $ runNet trained' rowP) validateRows
    let res'     = fmap (\(S1D' label, S1D' prediction) -> (maxIndex (SA.extract label), maxIndex (SA.extract prediction))) res
    print trained'
    putStrLn $ "Iteration " ++ show i ++ ": " ++ show (length (filter ((==) <$> fst <*> snd) res')) ++ " of " ++ show (length res')
    return trained'

data MnistOpts = MnistOpts FilePath FilePath Int LearningParameters

mnist' :: Parser MnistOpts
mnist' = MnistOpts <$> argument str (metavar "TRAIN")
                   <*> argument str (metavar "VALIDATE")
                   <*> option auto (long "iterations" <> short 'i' <> value 10)
                   <*> (LearningParameters
                       <$> option auto (long "train_rate" <> short 'r' <> value 0.01)
                       <*> option auto (long "momentum" <> value 0.9)
                       <*> option auto (long "l2" <> value 0.0005)
                       )

main :: IO ()
main = do
    MnistOpts mnist vali iter rate <- execParser (info (mnist' <**> helper) idm)
    putStrLn "Training convolutional neural network..."

    res <- runExceptT $ convTest iter mnist vali rate
    case res of
      Right () -> pure ()
      Left err -> putStrLn err

readMNIST :: FilePath -> ExceptT String IO [(S' ('D2 28 28), S' ('D1 10))]
readMNIST mnist = ExceptT $ do
  mnistdata <- T.readFile mnist
  return $ traverse (A.parseOnly parseMNIST) (T.lines mnistdata)

parseMNIST :: A.Parser (S' ('D2 28 28), S' ('D1 10))
parseMNIST = do
  lab     <- A.decimal
  pixels  <- many (A.char ',' >> A.double)
  let lab' = replicate lab 0 ++ [1] ++ replicate (9 - lab) 0
  return (S2D' $ SA.fromList pixels, S1D' $ SA.fromList lab')
