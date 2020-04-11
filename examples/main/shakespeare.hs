{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE CPP                   #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE TupleSections         #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE LambdaCase            #-}

import           Control.Monad.Random
import           Control.Monad.Trans.Except

import           Data.Char ( isUpper, toUpper, toLower )
import           Data.List ( foldl' )
import           Data.Maybe ( fromMaybe )

#if ! MIN_VERSION_base(4,13,0)
import           Data.Semigroup ( (<>) )
#endif

import qualified Data.Vector as V
import           Data.Vector ( Vector )

import qualified Data.Map as M
#if ! MIN_VERSION_base(4,13,0)
import           Data.Proxy ( Proxy (..) )
#endif

import qualified Data.ByteString as B
import           Data.Serialize

import           Data.Singletons.Prelude
import           GHC.TypeLits

import           Numeric.LinearAlgebra.Static ( konst )

import           Options.Applicative

import           Grenade
import           Grenade.Recurrent
import           Grenade.Utils.OneHot

import           System.IO.Unsafe ( unsafeInterleaveIO )

-- The defininition for our natural language recurrent network.
-- This network is able to learn and generate simple words in
-- about an hour.
--
-- Grab the input from
-- https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
--
-- This is a first class recurrent net.
--
-- The F and R types are tagging types to ensure that the runner and
-- creation function know how to treat the layers.
--
-- As an example, here's a short sequence generated.
--
-- > KING RICHARD III:
-- > And as the heaven her his words, we the son, I show sand stape but the lament to shall were the sons with a strend

type F = FeedForward
type R = Recurrent

-- The definition of our network
type Shakespeare = RecurrentNetwork '[ R (LSTM 40 80), R (LSTM 80 40), F (FullyConnected 40 40), F Logit]
                                    '[ 'D1 40, 'D1 80, 'D1 40, 'D1 40, 'D1 40 ]

-- The definition of the "sideways" input, which the network is fed recurrently.
type Shakespearian = RecurrentInputs  '[ R (LSTM 40 80), R (LSTM 80 40), F (FullyConnected 40 40), F Logit]

randomNet :: MonadRandom m => m Shakespeare
randomNet = randomRecurrent

-- | Load the data files and prepare a map of characters to a compressed int representation.
loadShakespeare :: FilePath -> ExceptT String IO (Vector Int, M.Map Char Int, Vector Char)
loadShakespeare path = do
  contents     <- lift $ readFile path
  let annotated = annotateCapitals contents
  (m,cs)       <- ExceptT . return $ hotMap (Proxy :: Proxy 40) annotated
  hot          <- ExceptT . return . note "Couldn't generate hot values" $ traverse (`M.lookup` m) annotated
  return (V.fromList hot, m, cs)

trainSlice :: LearningParameters -> Shakespeare -> Shakespearian -> Vector Int -> Int -> Int -> (Shakespeare, Shakespearian)
trainSlice !lrate !net !recIns input offset size =
  let e = fmap (x . oneHot) . V.toList $ V.slice offset size input
  in case reverse e of
    (o : l : xs) ->
      let examples = reverse $ (l, Just o) : ((,Nothing) <$> xs)
      in  trainRecurrent lrate net recIns examples
    _ -> error "Not enough input"
    where
      x = fromMaybe (error "Hot variable didn't fit.")

runShakespeare :: ShakespeareOpts -> ExceptT String IO ()
runShakespeare opts = do
  (shakespeare, oneHotMap, oneHotDictionary) <- loadShakespeare $ trainingFile opts
  (net0, i0) <- lift $
    case loadPath opts of
      Just loadFile -> netLoad loadFile
      Nothing -> (,0) <$> randomNet

  (trained, bestInput) <- lift $ foldM (\(!net, !io) size -> do
    xs <- take (iterations opts `div` 10) <$> getRandomRs (0, length shakespeare - size - 1)
    let (!trained, !bestInput) = foldl' (\(!n, !i) offset -> trainSlice (rate opts) n i shakespeare offset size) (net, io) xs
    results <- take 1000 <$> generateParagraph trained bestInput (temperature opts) oneHotMap oneHotDictionary ( S1D $ konst 0)
    putStrLn ("TRAINING STEP WITH SIZE: " ++ show size)
    putStrLn (unAnnotateCapitals results)
    return (trained, bestInput)
    ) (net0, i0) $ replicate 10 (sequenceSize opts)

  case savePath opts of
    Just saveFile -> lift . B.writeFile saveFile $ runPut (put trained >> put bestInput)
    Nothing -> return ()

generateParagraph :: forall layers shapes n a. (Last shapes ~ 'D1 n, Head shapes ~ 'D1 n, KnownNat n, Ord a)
  => RecurrentNetwork layers shapes
  -> RecurrentInputs layers
  -> Double
  -> M.Map a Int
  -> Vector a
  -> S ('D1 n)
  -> IO [a]
generateParagraph n s temp hotmap hotdict =
  go s
    where
  go x y =
    do let (_, ns, o) = runRecurrent n x y
       un            <- sample temp hotdict o
       Just re       <- return $ makeHot hotmap un
       rest          <- unsafeInterleaveIO $ go ns re
       return (un : rest)

data ShakespeareOpts = ShakespeareOpts {
    trainingFile :: FilePath
  , iterations   :: Int
  , rate         :: LearningParameters
  , sequenceSize :: Int
  , temperature  :: Double
  , loadPath     :: Maybe FilePath
  , savePath     :: Maybe FilePath
  }

shakespeare' :: Parser ShakespeareOpts
shakespeare' = ShakespeareOpts <$> argument str (metavar "TRAIN")
                               <*> option auto (long "examples" <> short 'e' <> value 1000000)
                               <*> (LearningParameters
                                    <$> option auto (long "train_rate" <> short 'r' <> value 0.01)
                                    <*> option auto (long "momentum" <> value 0.95)
                                    <*> option auto (long "l2" <> value 0.000001)
                                    )
                               <*> option auto (long "sequence-length" <> short 's' <> value 50)
                               <*> option auto (long "temperature" <> short 't' <> value 0.4)
                               <*> optional (strOption (long "load"))
                               <*> optional (strOption (long "save"))

main :: IO ()
main = do
    shopts <- execParser (info (shakespeare' <**> helper) idm)
    res <- runExceptT $ runShakespeare shopts
    case res of
      Right () -> pure ()
      Left err -> putStrLn err


netLoad :: FilePath -> IO (Shakespeare, Shakespearian)
netLoad modelPath = do
  modelData <- B.readFile modelPath
  either fail return $ runGet get modelData

-- Replace capitals with an annotation and the lower case letter
-- http://fastml.com/one-weird-trick-for-training-char-rnns/
annotateCapitals :: String -> String
annotateCapitals (x : rest)
    | isUpper x
    = '^' : toLower x : annotateCapitals rest
    | otherwise
    = x : annotateCapitals rest
annotateCapitals []
    = []

unAnnotateCapitals :: String -> String
unAnnotateCapitals ('^' : x : rest)
    = toUpper x : unAnnotateCapitals rest
unAnnotateCapitals (x : rest)
    =  x : unAnnotateCapitals rest
unAnnotateCapitals []
    = []

-- | Tag the 'Nothing' value of a 'Maybe'
note :: a -> Maybe b -> Either a b
note a = maybe (Left a) Right
