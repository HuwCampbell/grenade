{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE CPP                 #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TupleSections       #-}
{-# LANGUAGE TypeFamilies        #-}
{-# LANGUAGE TypeOperators       #-}

-- This is a simple generative adversarial network to make pictures
-- of numbers similar to those in MNIST.
--
-- It demonstrates a different usage of the library. Within about 15
-- minutes it was producing examples like this:
--
--               --.
--     .=-.--..#=###
--     -##==#########.
--     #############-
--   -###-.=..-.-==
--   ###-
--   .###-
--   .####...==-.
--   -####=--.=##=
--   -##=-     -##
--             =##
--           -##=
--           -###-
--         .####.
--         .#####.
-- ...---=#####-
-- .=#########.         .
--   .#######=.          .
--     . =-.
--
-- It's a 5!
--
import           Control.Applicative
import           Control.Monad
import           Control.Monad.Random
import           Control.Monad.Trans.Except

import qualified Data.Attoparsec.Text         as A
import qualified Data.ByteString              as B
import           Data.List                    (foldl')
#if ! MIN_VERSION_base(4,13,0)
import           Data.Semigroup               ((<>))
#endif
import qualified Data.Attoparsec.Text         as A
import qualified Data.ByteString              as B
import           Data.Semigroup               ((<>))
import           Data.Serialize
import qualified Data.Text                    as T
import qualified Data.Text.IO                 as T
import qualified Data.Vector.Storable         as V

import           Numeric.LinearAlgebra.Data   (toLists)
import qualified Numeric.LinearAlgebra.Static as SA

import           Options.Applicative

import           Grenade
import           Grenade.Utils.OneHot

type Discriminator =
  Network
    '[ Convolution 1 10 5 5 1 1, Pooling 2 2 2 2, Relu
     , Convolution 10 16 5 5 1 1, Pooling 2 2 2 2, Relu
     , Reshape, FullyConnected 256 80, Logit, FullyConnected 80 1, Logit]
    '[ 'D2 28 28
     , 'D3 24 24 10, 'D3 12 12 10, 'D3 12 12 10
     , 'D3 8 8 16, 'D3 4 4 16, 'D3 4 4 16
     , 'D1 256, 'D1 80, 'D1 80, 'D1 1, 'D1 1]

type Generator =
  Network
    '[ FullyConnected 80 256, Relu, Reshape
     , Deconvolution 16 10 5 5 2 2, Relu
     , Deconvolution 10 1 8 8 2 2, Logit]
    '[ 'D1 80
     , 'D1 256, 'D1 256, 'D3 4 4 16
     , 'D3 11 11 10, 'D3 11 11 10
     , 'D2 28 28, 'D2 28 28 ]

randomDiscriminator :: IO Discriminator
randomDiscriminator = randomNetwork

randomGenerator :: IO Generator
randomGenerator = randomNetwork

trainExample :: Optimizer opt -> Discriminator -> Generator -> S ('D2 28 28) -> S ('D1 80) -> ( Discriminator, Generator )
trainExample opt discriminator generator realExample noiseSource
 = let (generatorTape, fakeExample)       = runNetwork generator noiseSource

       (discriminatorTapeReal, guessReal) = runNetwork discriminator realExample
       (discriminatorTapeFake, guessFake) = runNetwork discriminator fakeExample

       (discriminator'real, _)            = runGradient discriminator discriminatorTapeReal ( guessReal - 1 )
       (discriminator'fake, _)            = runGradient discriminator discriminatorTapeFake guessFake
       (_, push)                          = runGradient discriminator discriminatorTapeFake ( guessFake - 1)

       (generator', _)                    = runGradient generator generatorTape push

       newDiscriminator                   = foldl' (applyUpdate $ sgdUpdateLearningParamters opt) discriminator [ discriminator'real, discriminator'fake ]
       newGenerator                       = applyUpdate opt generator generator'
   in ( newDiscriminator, newGenerator )
  where sgdUpdateLearningParamters :: Optimizer opt -> Optimizer opt
        sgdUpdateLearningParamters (OptSGD rate mom reg) = OptSGD rate mom (reg * 10)
        sgdUpdateLearningParamters o                     = o


ganTest :: (Discriminator, Generator) -> Int -> FilePath -> Optimizer opt -> ExceptT String IO (Discriminator, Generator)
ganTest (discriminator0, generator0) iterations trainFile opt = do
  trainData      <- fmap fst <$> readMNIST trainFile

  lift $ foldM (runIteration trainData) ( discriminator0, generator0 ) [1..iterations]

    where

  showShape' :: S ('D2 a b) -> IO ()
  showShape' (S2D mm) = putStrLn $
    let m  = SA.extract mm
        ms = toLists m
        render n'  | n' <= 0.2  = ' '
                   | n' <= 0.4  = '.'
                   | n' <= 0.6  = '-'
                   | n' <= 0.8  = '='
                   | otherwise =  '#'

        px = (fmap . fmap) render ms
    in unlines px

  runIteration :: [S ('D2 28 28)] -> (Discriminator, Generator) -> Int -> IO (Discriminator, Generator)
  runIteration trainData ( !discriminator, !generator ) _ = do
    trained'    <- foldM ( \(!discriminatorX, !generatorX ) realExample -> do
                      fakeExample <- randomOfShape
                      return $ trainExample opt discriminatorX generatorX realExample fakeExample
                     ) ( discriminator, generator ) trainData


    showShape' . snd . runNetwork (snd trained') =<< randomOfShape

    return trained'

data GanOpts = GanOpts FilePath Int Bool (Optimizer 'SGD) (Optimizer 'Adam) (Maybe FilePath) (Maybe FilePath)

mnist' :: Parser GanOpts
mnist' = GanOpts <$> argument str (metavar "TRAIN")
                 <*> option auto (long "iterations" <> short 'i' <> value 15)
                 <*> flag False True (long "use-adam" <> short 'a')
                 <*> (OptSGD
                       <$> option auto (long "train_rate" <> short 'r' <> value 0.01)
                       <*> option auto (long "momentum" <> value 0.9)
                       <*> option auto (long "l2" <> value 0.0005)
                       )
                 <*> (OptAdam
                       <$> option auto (long "alpha" <> short 'r' <> value 0.001)
                       <*> option auto (long "beta1" <> value 0.9)
                       <*> option auto (long "beta2" <> value 0.999)
                       <*> option auto (long "epsilon" <> value 1e-4)
                      )
                 <*> optional (strOption (long "load"))
                 <*> optional (strOption (long "save"))


main :: IO ()
main = do
  GanOpts mnist iter useAdam sgd adam load save <- execParser (info (mnist' <**> helper) idm)
  putStrLn "Training stupidly simply GAN"
  nets0 <-
    case load of
      Just loadFile -> netLoad loadFile
      Nothing       -> (,) <$> randomDiscriminator <*> randomGenerator
  res <-
    if useAdam
      then runExceptT $ ganTest nets0 iter mnist adam
      else runExceptT $ ganTest nets0 iter mnist sgd
  case res of
    Right nets1 ->
      case save of
        Just saveFile -> B.writeFile saveFile $ runPut (put nets1)
        Nothing       -> return ()
    Left err -> putStrLn err

readMNIST :: FilePath -> ExceptT String IO [(S ('D2 28 28), S ('D1 10))]
readMNIST mnist = ExceptT $ do
  mnistdata <- T.readFile mnist
  return $ traverse (A.parseOnly parseMNIST) (tail $ T.lines mnistdata)

parseMNIST :: A.Parser (S ('D2 28 28), S ('D1 10))
parseMNIST = do
  Just lab <- oneHot <$> A.decimal
  pixels   <- many (A.char ',' >> A.double)
  image    <- maybe (fail "Parsed row was of an incorrect size") pure (fromStorable . V.fromList $ pixels)
  return (image, lab)

netLoad :: FilePath -> IO (Discriminator, Generator)
netLoad modelPath = do
  modelData <- B.readFile modelPath
  either fail return $ runGet (get :: Get (Discriminator, Generator)) modelData
