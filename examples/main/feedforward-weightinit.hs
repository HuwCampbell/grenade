{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE CPP                 #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE ScopedTypeVariables #-}

import           Control.Monad
import           Control.Monad.Random
import           Data.Constraint                 (Dict (..))
import           Data.List                       (foldl')
import           Data.Serialize
import           Data.Singletons
import           Data.Singletons.Prelude.List
import qualified Data.Vector.Storable            as V
import           GHC.TypeLits
import qualified Numeric.LinearAlgebra.Static    as SA
import           Options.Applicative
import           System.Random.MWC
import           System.Random.MWC.Distributions
import           Unsafe.Coerce                   (unsafeCoerce)

import           Grenade

-- | Choose a backend: 0 BLAS, 1 HMATRIX
#define HMATRIX 0


netSpec :: SpecNet
netSpec = specFullyConnected 2 400 |=> specRelu1D 400 |=> specFullyConnected 400 1 |=> specLogit1D 1 |=> specNil1D 1

-- netSpec :: SpecNet
-- netSpec = specFullyConnected 2 4000 |=> specRelu1D 4000 |=> specFullyConnected 4000 1 |=> specLogit1D 1 |=> specNil1D 1


-- netSpec :: SpecNet
-- netSpec =
--       specFullyConnected 2 4500
--   |=> specTanh1D 4500
--   |=> specFullyConnected 4500 750
--   |=> specRelu1D 750
--   |=> specFullyConnected 750 150
--   |=> specRelu1D 150
--   |=> specFullyConnected 150 150
--   |=> specRelu1D 150
--   |=> specFullyConnected 150 30
--   |=> specRelu1D 30
--   |=> specFullyConnected 30 20
--   |=> specRelu1D 20
--   |=> specFullyConnected 20 10
--   |=> specRelu1D 10
--   |=> specFullyConnected 10 1
--   |=> specLogit1D 1
--   |=> specNil1D 1


-- -- -- | The definition for a feed forward network using the dynamic module. Note the nested networks. This network clearly is over-engeneered for this example!
-- netSpec :: SpecNet
-- netSpec =
--  specFullyConnected 2 40 |=> specTanh1D 40 |=>
--  specDropout 40 0.95 Nothing |=>
--  netSpecInner |=>
--  specFullyConnected 20 30 |=> specRelu1D 30 |=>
--  specFullyConnected 30 20 |=> specRelu1D 20 |=>
--  specFullyConnected 20 10 |=> specRelu1D 10 |=>
--  specFullyConnected 10 1 |=> specLogit1D 1 |=> specNil1D 1
--  where netSpecInner = specFullyConnected 40 30 |=> specRelu1D 30 |=> specFullyConnected 30 20 |=> specReshape1D2D 20 (2, 10) |=> specReshape2D1D (2, 10) 20 |=> specNil1D 20

goals :: [((RealNum, RealNum), RealNum)]
goals = [((0.50, 0.50), 0.25), ((-0.50, -0.50), 0.25)]


-- | Specifications can be built using the following interface also. Here one does not have to carefully pay attention to match the dimension inputs and outputs as when using specification directly.
buildNetViaInterface :: IO SpecConcreteNetwork
buildNetViaInterface =
  buildModelWith (NetworkInitSettings UniformInit HMatrix (Just 10)) (DynamicBuildSetup { printResultingSpecification = False })  $
  inputLayer1D 2 >>                            -- 1. Every model has to start with an input dimension
  fullyConnected 10 >> dropout 0.89 >> relu >> -- 2. Layers are simply added as desired. The input of each layer is determined automatically,
  fullyConnected 20 >> relu >>                 --    whereas the output is specified as in `fullyConnected 20`. Thus, the layer empits 20 signals
  fullyConnected 4 >> tanhLayer >>             --    which arethe input of the next layer.
  networkLayer (
    inputLayer1D 4 >> fullyConnected 10 >> relu >> fullyConnected 4 >> sinusoid
    ) >>
  reshape (4, 1, 1) >>                         -- 3. Reshape is ignored if the input and output is the same
  fullyConnected 1                             -- 4. The output is simply the number of signals the alst layer emits.


netTrain :: forall shapes len1 len2 layers o .
     (SingI (Last shapes), KnownNat len1, KnownNat len2, Head shapes ~ 'D1 len1, Last shapes ~ 'D1 len2)
  => Network layers shapes
  -> Optimizer o
  -> Int
  -> IO (Network layers shapes)
netTrain net0 op n = do
#if HMATRIX
    inps <- replicateM n $ do
      s  <- getRandom
      return $ S1D (SA.randomVector s SA.Uniform * 2 - 1 :: SA.R len1)
#else
    inps <- replicateM n $ do
        s  <- getRandom
        return $ S1DV $ SA.extract (SA.randomVector s SA.Uniform * 2 - 1 :: SA.R len1)
#endif
    let mkVal :: S ('D1 len1) -> Rational
        mkVal (S1D v) = mkVal (S1DV $ SA.extract v)
        mkVal (S1DV v) = if any (v `inCircleV`) goals
                         then 1
                         else 0
#if HMATRIX
        constr = S1D . fromRational
#else
        constr = S1DV . fromRational
#endif
    let outs = map (constr . mkVal) inps
    let trained = foldl' trainEach net0 (zip inps outs)
    return trained

  where trainEach !network (i,o) = train op network i o

netScore :: (KnownNat len, Head shapes ~ 'D1 len, Last shapes ~ 'D1 1) => Network layers shapes -> IO ()
netScore network = do
    let testIns = [ [ (x,y)  | x <- [0..50] ]
                             | y <- [0..20] ]
#if HMATRIX
        outMat  = fmap (fmap (\(x,y) -> (render (x/25-1) (y/10-1) . normx) $ runNet network (S1D $ SA.vector [x / 25 - 1,y / 10 - 1]))) testIns
#else
        outMat  = fmap (fmap (\(x,y) -> (render (x/25-1) (y/10-1) . normx) $ runNet network (S1DV $ V.fromList [x / 25 - 1,y / 10 - 1]))) testIns
#endif
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

normx :: S ('D1 1) -> RealNum
normx (S1D r)  = SA.mean r
normx (S1DV v) = V.sum v / fromIntegral (V.length v)

testValues :: forall len shapes layers . (KnownNat len, Head shapes ~ 'D1 len, Last shapes ~ 'D1 1) => Network layers shapes -> IO ()
testValues network = do
  let n = 1000
#if HMATRIX
  inps <- replicateM n $ do
      s  <- getRandom
      return $ S1D (SA.randomVector s SA.Uniform * 2 - 1 :: SA.R len)
#else
  inps <- replicateM n $ do
       s  <- getRandom
       return $ S1DV $ SA.extract (SA.randomVector s SA.Uniform * 2 - 1 :: SA.R len)
#endif

  let mkVal :: S ('D1 len1) -> Integer
      mkVal (S1D v) = mkVal (S1DV $ SA.extract v :: S ('D1 len))
      mkVal (S1DV v) = if any (v `inCircleV`) goals
                         then 1
                         else 0
  let outs :: [Integer]
      outs = map mkVal inps
  let ress = zip outs (map (round . normx . runNet network) inps)
      correct = length $ filter id $ map (uncurry (==)) ress
      incorrect = length $ filter id $ map (uncurry (/=)) ress
      falsePositives = length $ filter id $ map (uncurry (\shd nn -> shd == 0 && nn == 1)) ress
      falseNegatives = length $ filter id $ map (uncurry (\shd nn -> shd == 1 && nn == 0)) ress
  putStr $ show correct  ++ " | "
  putStr $ show incorrect ++ " | "
  putStr $ show falsePositives ++ " | "
  putStrLn $ show falseNegatives ++ " | "


inCircle :: KnownNat n => SA.R n -> (SA.R n, RealNum) -> Bool
v `inCircle` (o, r) = SA.norm_2 (v - o) <= r

inCircleV :: V.Vector RealNum -> ((RealNum, RealNum), RealNum) -> Bool
v `inCircleV` ((x,y), r) = norm (V.zipWith (-) v circle) <= r
  where
    circle = V.fromList [x,y]
    norm x = sqrt (V.sum $ V.map (^ (2 :: Int)) x)


data FeedForwardOpts = FeedForwardOpts Int (Optimizer 'Adam)

feedForward' :: Parser FeedForwardOpts
feedForward' =
  FeedForwardOpts <$> option auto (long "examples" <> short 'e' <> value 10000)
                  <*> (OptAdam
                       <$> option auto (long "alpha" <> short 'r' <> value 0.001)
                       <*> option auto (long "beta1" <> value 0.9)
                       <*> option auto (long "beta2" <> value 0.999)
                       <*> option auto (long "epsilon" <> value 1e-4)
                       <*> option auto (long "lambda" <> value 1e-3))


main :: IO ()
main = do


  FeedForwardOpts examples rate <- execParser (info (feedForward' <**> helper) idm)
  putStrLn "| Nr | Correct | Incorrect | FalsePositives | FalseNegatives |"
  putStrLn "--------------------------------------------------------------"
  let nr = 10 :: Int
  mapM_
    (\n -> do
       putStr $ "| " ++ show n ++ " | "
#if HMATRIX
       SpecConcreteNetwork1D1D (net0 :: Network layers shapes) <- networkFromSpecificationWith (NetworkInitSettings HeEtAl HMatrix Nothing) netSpec
#else
       SpecConcreteNetwork1D1D (net0 :: Network layers shapes) <- networkFromSpecificationWith (NetworkInitSettings HeEtAl BLAS Nothing) netSpec
#endif
       -- We need to specify the actual number of output nodes, as our functions requiere that!
       case (unsafeCoerce (Dict :: Dict ()) :: Dict (('D1 1) ~ Last shapes)) of
         Dict -> do
           net <- netTrain net0 rate examples
           unsafeCoerce $ -- only needed as GADTs are enabled, which disallowes the type to escape and thus prevents the type inference to work. The result is not needed anyways.
             testValues net)
    [1 .. nr]


  -- Features of dynamic networks:
  SpecConcreteNetwork1D1D (net' :: Network layers shapes) <- networkFromSpecificationWith (NetworkInitSettings HeEtAl BLAS (Just 50000)) netSpec
  net2 <- netTrain net' rate 0
  let spec' = networkToSpecification net2
  putStrLn "String represenation of the network specification: "
  print spec'
  let serializedSpec = encode spec'   -- only the specification (not the weights) are serialized here! The weights can be serialized using the networks serialize instance!
  let _ = encode net2                 -- E.g. like this.
  case decode serializedSpec of
    Left err -> print err
    Right spec2 -> do
      print spec2
      SpecConcreteNetwork1D1D (net3 :: Network layers3 shapes3) <- networkFromSpecificationWith (NetworkInitSettings HeEtAl BLAS (Just 50000)) spec2
      net4 <- foldM (\n _ -> netTrain n rate examples) net3 [(1 :: Int)..30]
      case (unsafeCoerce (Dict :: Dict ()) :: Dict (('D1 1) ~ Last shapes3)) of
        Dict -> netScore net4

  -- -- There is a also nice interface available also
  -- SpecConcreteNetwork1D1D newNet <- buildNetViaInterface
  -- print (networkToSpecification newNet)
