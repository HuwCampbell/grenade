{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE CPP                   #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE TupleSections         #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE FlexibleContexts      #-}

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

import           Numeric.LinearAlgebra ( maxIndex )
import qualified Numeric.LinearAlgebra.Static as SA

import           Options.Applicative
import           System.FilePath ( (</>) )

import           Grenade
import           Grenade.Utils.OneHot

-- It's logistic regression!
--
-- This network is used to show how we can embed a Network as a layer in the larger MNIST
-- type.
type FL i o =
  Network
    '[ FullyConnected i o, Logit ]
    '[ 'D1 i, 'D1 o, 'D1 o ]

-- The definition of our convolutional neural network.
-- In the type signature, we have a type level list of shapes which are passed between the layers.
-- One can see that the images we are inputing are two dimensional with 28 * 28 pixels.

-- It's important to keep the type signatures, as there's many layers which can "squeeze" into the gaps
-- between the shapes, so inference can't do it all for us.

-- With the mnist data from Kaggle normalised to doubles between 0 and 1, learning rate of 0.01 and 15 iterations,
-- this network should get down to about a 1.3% error rate.
--
-- /NOTE:/ This model is actually too complex for MNIST, and one should use the type given in the readme instead.
--         This one is just here to demonstrate Inception layers in use.
--
type MNIST =
  Network
    '[ Reshape,
       Concat ('D3 28 28 1) Trivial ('D3 28 28 14) (InceptionMini 28 28 1 5 9),
       Pooling 2 2 2 2, Relu,
       Concat ('D3 14 14 3) (Convolution 15 3 1 1 1 1) ('D3 14 14 15) (InceptionMini 14 14 15 5 10), Crop 1 1 1 1, Pooling 3 3 3 3, Relu,
       Reshape, FL 288 80, FL 80 10 ]
    '[ 'D2 28 28, 'D3 28 28 1,
       'D3 28 28 15, 'D3 14 14 15, 'D3 14 14 15, 'D3 14 14 18,
       'D3 12 12 18, 'D3 4 4 18, 'D3 4 4 18,
       'D1 288, 'D1 80, 'D1 10 ]

randomMnist :: MonadRandom m => m MNIST
randomMnist = randomNetwork

<<<<<<< HEAD
convTest :: Int -> FilePath -> Maybe Int -> LearningParameters -> IO ()
convTest iterations dataDir nSamples rate = do
  net0         <- randomMnist
  trainData    <- readMNIST (dataDir </> "train-images-idx3-ubyte.gz")
                            (dataDir </> "train-labels-idx1-ubyte.gz")
  validateData <- readMNIST (dataDir </> "t10k-images-idx3-ubyte.gz")
                            (dataDir </> "t10k-labels-idx1-ubyte.gz")

  foldM_ (runIteration (maybe trainData (`take` trainData) nSamples) validateData) net0 [1..iterations]
=======
convTest :: Int -> FilePath -> LearningParameters -> IO ()
convTest iterations dataDir rate = do
  net0      <- randomMnist
  trainData <- readMNIST (mkFilePath "train-images-idx3-ubyte.gz")
                         (mkFilePath "train-labels-idx1-ubyte.gz")

  validateData <- readMNIST (mkFilePath "t10k-images-idx3-ubyte.gz")
                            (mkFilePath "t10k-labels-idx1-ubyte.gz")
  foldM_ (runIteration trainData validateData) net0 [1 .. iterations]

 where

  mkFilePath :: String -> FilePath
  mkFilePath s = dataDir ++ '/' : s
>>>>>>> Work in progress

  trainEach rate' !network (i, o) = train rate' network i o

  runIteration trainRows validateRows net i = do
<<<<<<< HEAD
    let trained' = foldl' (trainEach ( rate { learningRate = learningRate rate * 0.9 ^ i} )) net trainRows
    print trained'

    putStrLn "Checking..."
    let res      = fmap (\(rowP,rowL) -> (rowL,) $ runNet trained' rowP) validateRows
    let res'     = fmap (\(S1D label, S1D prediction) -> (maxIndex (SA.extract label), maxIndex (SA.extract prediction))) res
    let matched   = length $ filter ((==) <$> fst <*> snd) res'
    let total     = length res'
    let matchedpc = fromIntegral matched / fromIntegral total * 100.0 :: Float
    putStrLn $ "Iteration " ++ show i ++ ": matched " ++ show matched ++ " of " ++ show total ++ " (" ++ show matchedpc ++ "%)" 
    return trained'

data MnistOpts = MnistOpts FilePath (Maybe Int) Int LearningParameters

mnist' :: Parser MnistOpts
mnist' = MnistOpts <$> argument str (metavar "DATADIR")
                       -- option to reduce the number of training samples used from 60,000
                       -- to avoid running out of memory
                   <*> option (Just <$> auto) (long "limit_samples_to" <> short 'l' <> value Nothing)
                   <*> option auto (long "iterations" <> short 'i' <> value 15)
                   <*> (LearningParameters
                       <$> option auto (long "train_rate" <> short 'r' <> value 0.01)
                       <*> option auto (long "momentum" <> value 0.9)
                       <*> option auto (long "l2" <> value 0.0005)
                       )

main :: IO ()
main = do
    MnistOpts dataDir nSamples iter rate <- execParser (info (mnist' <**> helper) idm)
    putStr "Training convolutional neural network with "
    putStr $ maybe "all" show nSamples
    putStrLn " samples..."

    convTest iter dataDir nSamples rate


-- Adapted from https://github.com/tensorflow/haskell/blob/master/tensorflow-mnist/src/TensorFlow/Examples/MNIST/Parse.hs
-- Could also have used Data.IDX, although that uses a different Vector variant from that need for fromStorable
readMNIST :: FilePath -> FilePath -> IO [(S ( 'D2 28 28), S ( 'D1 10))]
readMNIST iFP lFP = do
  labels  <- readMNISTLabels lFP
  samples <- readMNISTSamples iFP
  return $ zip
    (fmap (fromMaybe (error "bad samples") . fromStorable) samples)
    (fromMaybe (error "bad labels") . oneHot . fromIntegral <$> labels)
=======
    let trained' = foldl'
          (trainEach (rate { learningRate = learningRate rate * 0.9 ^ i }))
          net
          trainRows
    let res =
          fmap (\(rowP, rowL) -> (rowL, ) $ runNet trained' rowP) validateRows
    let res' = fmap
          (\(S1D label, S1D prediction) ->
            (maxIndex (SA.extract label), maxIndex (SA.extract prediction))
          )
          res
    print trained'
    putStrLn
      $  "Iteration "
      ++ show i
      ++ ": "
      ++ show (length (filter ((==) <$> fst <*> snd) res'))
      ++ " of "
      ++ show (length res')
    return trained'

data MnistOpts = MnistOpts FilePath Int LearningParameters

mnist' :: Parser MnistOpts
mnist' =
  MnistOpts
    <$> argument str (metavar "DATADIR")
    <*> option auto (long "iterations" <> short 'i' <> value 15)
    <*> (   LearningParameters
        <$> option auto (long "train_rate" <> short 'r' <> value 0.01)
        <*> option auto (long "momentum" <> value 0.9)
        <*> option auto (long "l2" <> value 0.0005)
        )

main :: IO ()
main = do
  MnistOpts mnistDir iter rate <- execParser (info (mnist' <**> helper) idm)
  putStrLn "Training convolutional neural network..."

  convTest iter mnistDir rate

-- Adapted from https://github.com/tensorflow/haskell/blob/master/tensorflow-mnist/src/TensorFlow/Examples/MNIST/Parse.hs
-- Could also have used Data.IDX, although that uses a different Vector variant from that need for fromStorable

readMNIST :: FilePath -> FilePath -> IO [(S ( 'D2 28 28), S ( 'D1 10))]
readMNIST iFP lFP = do
  labels  <- readMNISTLabels lFP
  samples <- readMNISTSamples iFP
  return $ zip
    (fmap (fromMaybe (error "bad samples") . fromStorable) samples)
    ((fromMaybe (error "bad labels") . oneHot . fromIntegral) <$> labels)

<<<<<<< HEAD
loadMNIST :: FilePath -> FilePath -> IO (Maybe [(S ( 'D2 28 28), S ( 'D1 10))])
loadMNIST iFP lFP =  do
  let labels  = MaybeT $ readMNISTLabels lFP
  let samples = MaybeT $ fromIntegral . readMNISTSamples $ iFP
  d <- MaybeT (fromStorable samples, (oneHot . fromIntegral) <$> labels)
  return d
>>>>>>> Work in progress
=======
>>>>>>> Adapt MNIST example to use original data files

-- | Check's the file's endianess, throwing an error if it's not as expected.
checkEndian :: Get ()
checkEndian = do
<<<<<<< HEAD
  magic <- Serialize.getWord32be
  when (magic `notElem` ([2049, 2051] :: [Word32]))
    $ error "Expected big endian, but image file is little endian."

-- | Reads an MNIST file and returns a list of samples.
readMNISTSamples :: FilePath -> IO [V.Vector Double]
readMNISTSamples path = do
  raw <- decompress <$> B.readFile path
  either fail ( return . fmap (V.map normalize) ) $ Serialize.runGetLazy getMNIST raw
=======
  magic <- getWord32be
  when (magic `notElem` ([2049, 2051] :: [Word32]))
    $ fail "Expected big endian, but image file is little endian."

-- | Reads an MNIST file and returns a list of samples.
readMNISTSamples :: FilePath -> IO [V.Vector Double]
readMNISTSamples path = do
  raw <- decompress <$> BS.readFile path
<<<<<<< HEAD
  return $ runGet getMNIST raw
>>>>>>> Work in progress
=======
  return . fmap (V.map normalize) $ runGet getMNIST raw
>>>>>>> Adapt MNIST example to use original data files
 where
  getMNIST :: Get [V.Vector Word8]
  getMNIST = do
    checkEndian
    -- Parse header data.
<<<<<<< HEAD
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
=======
    cnt    <- fromIntegral <$> getWord32be
    rows   <- fromIntegral <$> getWord32be
    cols   <- fromIntegral <$> getWord32be
    -- Read all of the data, then split into samples.
    pixels <- getLazyByteString $ fromIntegral $ cnt * rows * cols
    return $ V.fromList <$> chunksOf (rows * cols) (BS.unpack pixels)
>>>>>>> Work in progress

  normalize :: Word8 -> Double
  normalize = (/ 255) . fromIntegral
  --normalize = (/ 0.3081) . (`subtract` 0.1307) . (/ 255) . fromIntegral

-- | Reads a list of MNIST labels from a file and returns them.
readMNISTLabels :: FilePath -> IO [Word8]
readMNISTLabels path = do
<<<<<<< HEAD
  raw <- decompress <$> B.readFile path
  either fail return $ Serialize.runGetLazy getLabels raw
=======
  raw <- decompress <$> BS.readFile path
  return $ runGet getLabels raw
>>>>>>> Work in progress
 where
  getLabels :: Get [Word8]
  getLabels = do
    checkEndian
    -- Parse header data.
<<<<<<< HEAD
    cnt <- fromIntegral <$> Serialize.getWord32be
    -- Read all of the labels.
    B.unpack <$> Serialize.getLazyByteString cnt
=======
    cnt <- fromIntegral <$> getWord32be
    -- Read all of the labels.
    BS.unpack <$> getLazyByteString cnt
>>>>>>> Work in progress

