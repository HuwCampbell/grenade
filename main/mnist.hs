{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE KindSignatures        #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE TupleSections         #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE FlexibleContexts      #-}

import           Control.Applicative
import           Control.Monad
import           Control.Monad.Identity
import           Control.Monad.Random

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
randomMnistNet :: (MonadRandom m) => m (Network Identity '[('D2 28 28), ('D2 32 32), ('D3 28 28 10), ('D3 14 14 10), ('D3 14 14 10), ('D3 10 10 16), ('D3 5 5 16), ('D1 400), ('D1 400), ('D1 80), ('D1 80), ('D1 10), ('D1 10)])
randomMnistNet = do
  let pad :: Pad 2 2 2 2          = Pad
  a :: Convolution 1 10 5 5 1 1  <- randomConvolution
  let b :: Pooling 2 2 2 2        = Pooling
  c :: Convolution 10 16 5 5 1 1 <- randomConvolution
  let d :: Pooling 2 2 2 2        = Pooling
  e :: FullyConnected 400 80     <- randomFullyConnected
  f :: FullyConnected 80  10     <- randomFullyConnected
  return $ pad :~> a :~> b :~> Relu :~> c :~> d :~> FlattenLayer :~> Relu :~> e :~> Logit :~> f :~> O Logit

convTest :: Int -> FilePath -> FilePath -> Double -> IO ()
convTest iterations trainFile validateFile rate = do
  net0 <- evalRandIO randomMnistNet
  fT   <- T.readFile trainFile
  fV   <- T.readFile validateFile
  let trainRows = traverse (A.parseOnly p) (T.lines fT)
  let validateRows = traverse (A.parseOnly p) (T.lines fV)
  case (trainRows, validateRows) of
    (Right tr', Right vr') -> foldM_ (runIteration tr' vr') net0 [1..iterations]
    err                    -> putStrLn $ show err

  where
    trainEach !rate' !nt !(i, o) = train rate' i o nt

    p :: A.Parser (S' ('D2 28 28), S' ('D1 10))
    p = do
      lab     <- A.decimal
      pixels  <- many (A.char ',' >> A.double)
      let lab' = replicate lab 0 ++ [1] ++ replicate (9 - lab) 0
      return (S2D' $ SA.fromList pixels, S1D' $ SA.fromList lab')

    runIteration trainRows validateRows net i = do
      let trained' = runIdentity $ foldM (trainEach (rate * (0.9 ^ i))) net trainRows
      let res      = runIdentity $ traverse (\(rowP,rowL) -> (rowL,) <$> runNet trained' rowP) validateRows
      let res'     = fmap (\(S1D' label, S1D' prediction) -> (maxIndex (SA.extract label), maxIndex (SA.extract prediction))) res
      putStrLn $ show trained'
      putStrLn $ "Iteration " ++ show i ++ ": " ++ show (length (filter ((==) <$> fst <*> snd) res')) ++ " of " ++ show (length res')
      return trained'

data MnistOpts = MnistOpts FilePath FilePath Int Double

mnist' :: Parser MnistOpts
mnist' = MnistOpts <$> (argument str (metavar "TRAIN"))
                   <*> (argument str (metavar "VALIDATE"))
                   <*> option auto (long "iterations" <> short 'i' <> value 15)
                   <*> option auto (long "train_rate" <> short 'r' <> value 0.01)

main :: IO ()
main = do
    MnistOpts mnist vali iter rate <- execParser (info (mnist' <**> helper) idm)
    putStrLn "Training convolutional neural network..."
    convTest iter mnist vali rate
