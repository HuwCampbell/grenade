{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE CPP                   #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE FlexibleContexts      #-}

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

import           Codec.Compression.GZip ( decompress )
import           Data.Serialize ( Get )
import qualified Data.Serialize as Serialize
import qualified Data.ByteString.Lazy as B

import           Data.List ( foldl' )
import           Data.List.Split ( chunksOf )
import           Data.Maybe ( fromMaybe )
#if ! MIN_VERSION_base(4,13,0)
import           Data.Semigroup ( (<>) )
#endif

import           Data.Word ( Word32 , Word8 )
import qualified Data.Vector.Storable as V

import qualified Numeric.LinearAlgebra.Static as SA
import           Numeric.LinearAlgebra.Data ( toLists )

import           Options.Applicative
import           System.FilePath ( (</>) ) 

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

randomDiscriminator :: MonadRandom m => m Discriminator
randomDiscriminator = randomNetwork

randomGenerator :: MonadRandom m => m Generator
randomGenerator = randomNetwork

trainExample :: LearningParameters -> Discriminator -> Generator -> S ('D2 28 28) -> S ('D1 80) -> ( Discriminator, Generator )
trainExample rate discriminator generator realExample noiseSource
 = let (generatorTape, fakeExample)       = runNetwork generator noiseSource

       (discriminatorTapeReal, guessReal) = runNetwork discriminator realExample
       (discriminatorTapeFake, guessFake) = runNetwork discriminator fakeExample

       (discriminator'real, _)            = runGradient discriminator discriminatorTapeReal ( guessReal - 1 )
       (discriminator'fake, _)            = runGradient discriminator discriminatorTapeFake guessFake
       (_, push)                          = runGradient discriminator discriminatorTapeFake ( guessFake - 1)

       (generator', _)                    = runGradient generator generatorTape push

       newDiscriminator                   = foldl' (applyUpdate rate { learningRegulariser = learningRegulariser rate * 10}) discriminator [ discriminator'real, discriminator'fake ]
       newGenerator                       = applyUpdate rate generator generator'
   in ( newDiscriminator, newGenerator )


ganTest :: (Discriminator, Generator) -> Int -> FilePath -> LearningParameters -> IO (Discriminator, Generator)
ganTest (discriminator0, generator0) iterations dataDir rate = do
  -- Note that for this example we use only the samples, and not the labels
  trainData      <- fmap fst <$> readMNIST (dataDir </> "train-images-idx3-ubyte.gz")
                                           (dataDir </> "train-labels-idx1-ubyte.gz")

  foldM (runIteration trainData) ( discriminator0, generator0 ) [1..iterations]

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
    trained'    <- foldM ( \(!discriminatorX, !generatorX ) realExample ->
                             trainExample rate discriminatorX generatorX realExample <$> randomOfShape )
                         ( discriminator, generator ) trainData


    showShape' . snd . runNetwork (snd trained') =<< randomOfShape

    return trained'

data GanOpts = GanOpts FilePath Int LearningParameters (Maybe FilePath) (Maybe FilePath)

mnist' :: Parser GanOpts
mnist' = GanOpts <$> argument str (metavar "DATADIR")
                 <*> option auto (long "iterations" <> short 'i' <> value 15)
                 <*> (LearningParameters
                       <$> option auto (long "train_rate" <> short 'r' <> value 0.01)
                       <*> option auto (long "momentum" <> value 0.9)
                       <*> option auto (long "l2" <> value 0.0005)
                       )
                 <*> optional (strOption (long "load"))
                 <*> optional (strOption (long "save"))


main :: IO ()
main = do
  GanOpts mnist iter rate load save <- execParser (info (mnist' <**> helper) idm)
  putStrLn "Training stupidly simply GAN"
  nets0 <- case load of
    Just loadFile -> netLoad loadFile
    Nothing -> (,) <$> randomDiscriminator <*> randomGenerator

  nets1 <- ganTest nets0 iter mnist rate
  case save of
    Just saveFile -> B.writeFile saveFile $ Serialize.runPutLazy (Serialize.put nets1)
    Nothing -> return ()


-- Adapted from https://github.com/tensorflow/haskell/blob/master/tensorflow-mnist/src/TensorFlow/Examples/MNIST/Parse.hs
-- Could also have used Data.IDX, although that uses a different Vector variant from that need for fromStorable
readMNIST :: FilePath -> FilePath -> IO [(S ( 'D2 28 28), S ( 'D1 10))]
readMNIST iFP lFP = do
  labels  <- readMNISTLabels lFP
  samples <- readMNISTSamples iFP
  return $ zip
    (fmap (fromMaybe (error "bad samples") . fromStorable) samples)
    (fromMaybe (error "bad labels") . oneHot . fromIntegral <$> labels)

-- | Check's the file's endianess, throwing an error if it's not as expected.
checkEndian :: Get ()
checkEndian = do
  magic <- Serialize.getWord32be
  when (magic `notElem` ([2049, 2051] :: [Word32]))
    $ error "Expected big endian, but image file is little endian."

-- | Reads an MNIST file and returns a list of samples.
readMNISTSamples :: FilePath -> IO [V.Vector Double]
readMNISTSamples path = do
  raw <- decompress <$> B.readFile path
  either fail ( return . fmap (V.map normalize) ) $ Serialize.runGetLazy getMNIST raw
 where
  getMNIST :: Get [V.Vector Word8]
  getMNIST = do
    checkEndian
    -- Parse header data.
    cnt    <- fromIntegral <$> Serialize.getWord32be
    rows   <- fromIntegral <$> Serialize.getWord32be
    cols   <- fromIntegral <$> Serialize.getWord32be
    -- Read all of the data, then split into samples.
    pixels <- Serialize.getLazyByteString $ fromIntegral $ cnt * rows * cols
    return $ V.fromList <$> chunksOf (rows * cols) (B.unpack pixels)

  normalize :: Word8 -> Double
  normalize = (/ 255) . fromIntegral
  -- There are other normalization functions in the literature, such as
  -- normalize = (/ 0.3081) . (`subtract` 0.1307) . (/ 255) . fromIntegral
  -- but we need values in the range [0..1] for the showShape' pretty printer

-- | Reads a list of MNIST labels from a file and returns them.
readMNISTLabels :: FilePath -> IO [Word8]
readMNISTLabels path = do
  raw <- decompress <$> B.readFile path
  either fail return $ Serialize.runGetLazy getLabels raw
 where
  getLabels :: Get [Word8]
  getLabels = do
    checkEndian
    -- Parse header data.
    cnt <- fromIntegral <$> Serialize.getWord32be
    -- Read all of the labels.
    B.unpack <$> Serialize.getLazyByteString cnt


netLoad :: FilePath -> IO (Discriminator, Generator)
netLoad modelPath = do
  modelData <- B.readFile modelPath
  either fail return $
    Serialize.runGetLazy (Serialize.get :: Get (Discriminator, Generator)) modelData
