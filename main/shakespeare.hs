{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE RecordWildCards       #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE TupleSections         #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE LambdaCase            #-}

import           Control.Monad.Random
import           Control.Monad.Trans.Except

import           Data.Char ( isUpper, toUpper, toLower )
import           Data.List ( unfoldr, foldl' )
import           Data.Maybe ( fromMaybe )

import qualified Data.Vector as V
import           Data.Vector ( Vector )

import qualified Data.Map as M
import           Data.Proxy ( Proxy (..) )


import           Data.Singletons.Prelude
import           GHC.TypeLits

import           Numeric.LinearAlgebra.Static ( konst )

import           Options.Applicative

import           Grenade
import           Grenade.Recurrent
import           Grenade.Utils.OneHot

-- The defininition for our natural language recurrent network.
-- This network is able to learn and generate simple words in
-- about an hour.
--
-- This is a first class recurrent net, although it's similar to
-- an unrolled graph.
--
-- The F and R types are tagging types to ensure that the runner and
-- creation function know how to treat the layers.
--
-- As an example, here's a short sequence generated.
--
-- > the see and and the sir, and and the make and the make and go the make and go the make and the
--
type F = FeedForward
type R = Recurrent

-- The definition of our network
type Shakespeare = RecurrentNetwork '[ R (LSTM 40 40), F (FullyConnected 40 40), F Logit]
                                    '[ 'D1 40, 'D1 40, 'D1 40, 'D1 40 ]

-- The definition of the "sideways" input, which the network if fed recurrently.
type Shakespearian = RecurrentInputs  '[ R (LSTM 40 40), F (FullyConnected 40 40), F Logit]

randomNet :: MonadRandom m => m (Shakespeare, Shakespearian)
randomNet = randomRecurrent

-- | Load the data files and prepare a map of characters to a compressed int representation.
loadShakespeare :: FilePath -> ExceptT String IO (Vector Int, M.Map Char Int, Vector Char)
loadShakespeare path = do
  contents     <- lift $ readFile path
  let annotated = annotateCapitals contents
  (m,cs)       <- ExceptT . return . note "Couldn't fit data in hotMap" $ hotMap (Proxy :: Proxy 40) annotated
  hot          <- ExceptT . return . note "Couldn't generate hot values" $ traverse (`M.lookup` m) annotated
  return (V.fromList hot, m, cs)

trainSlice :: LearningParameters -> Shakespeare -> Shakespearian -> Vector Int -> Int -> Int -> (Shakespeare, Shakespearian)
trainSlice !rate !net !recIns input offset size =
  let e = fmap (x . oneHot) . V.toList $ V.slice offset size input
  in case reverse e of
    (o : l : xs) ->
      let examples = reverse $ (l, Just o) : ((,Nothing) <$> xs)
      in  trainRecurrent rate net recIns examples
    _ -> error "Not enough input"
    where
      x = fromMaybe (error "Hot variable didn't fit.")

runShakespeare :: ShakespeareOpts -> ExceptT String IO ()
runShakespeare ShakespeareOpts {..} = do
  (shakespeare, oneHotMap, oneHotDictionary) <- loadShakespeare trainingFile
  (net0, i0) <- lift randomNet
  lift $ foldM_ (\(!net, !io) size -> do
    xs <- take (iterations `div` 15) <$> getRandomRs (0, length shakespeare - size - 1)
    let (!trained, !bestInput) = foldl' (\(!n, !i) offset -> trainSlice rate n i shakespeare offset size) (net, io) xs
    let results = take 100 $ generateParagraph trained bestInput oneHotMap oneHotDictionary ( S1D $ konst 0)
    putStrLn ("TRAINING STEP WITH SIZE: " ++ show size)
    putStrLn (unAnnotateCapitals results)
    return (trained, bestInput)
    ) (net0, i0) [10,10,15,15,20,20,25,25,30,30,35,35,40,40,50 :: Int]

generateParagraph :: forall layers shapes n a. (Last shapes ~ 'D1 n, Head shapes ~ 'D1 n, KnownNat n, Ord a)
  => RecurrentNetwork layers shapes
  -> RecurrentInputs layers
  -> M.Map a Int
  -> Vector a
  -> S ('D1 n)
  -> [a]
generateParagraph n s hotmap hotdict i =
  unfoldr go (s, i)
    where
  go (x, y) =
    do let (ns, o) = runRecurrent n x y
       un         <- unHot hotdict o
       re         <- makeHot hotmap un
       Just (un, (ns, re))

data ShakespeareOpts = ShakespeareOpts {
    trainingFile :: FilePath
  , iterations   :: Int
  , rate         :: LearningParameters
  }

shakespeare' :: Parser ShakespeareOpts
shakespeare' = ShakespeareOpts <$> argument str (metavar "TRAIN")
                               <*> option auto (long "examples" <> short 'e' <> value 1000000)
                               <*> (LearningParameters
                                    <$> option auto (long "train_rate" <> short 'r' <> value 0.01)
                                    <*> option auto (long "momentum" <> value 0.95)
                                    <*> option auto (long "l2" <> value 0.000001)
                                    )

main :: IO ()
main = do
    shopts <- execParser (info (shakespeare' <**> helper) idm)
    res <- runExceptT $ runShakespeare shopts
    case res of
      Right () -> pure ()
      Left err -> putStrLn err


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
