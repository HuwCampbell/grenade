{-# LANGUAGE BangPatterns        #-}
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


-- netSpec :: SpecNet
-- netSpec = specFullyConnected 2 400 |=> specRelu1D 400 |=> specFullyConnected 400 1 |=> specLogit1D 1 |=> specNil1D 1

-- -- | The definition for a feed forward network using the dynamic module. Note the nested networks. This network clearly is over-engeneered for this example!
netSpec :: SpecNet
netSpec =
  specFullyConnected 2 40 |=> specTanh1D 40 |=>
  specDropout 40 0.95 Nothing |=>
  netSpecInner |=>
  specFullyConnected 20 30 |=> specRelu1D 30 |=>
  specFullyConnected 30 20 |=> specRelu1D 20 |=>
  specFullyConnected 20 10 |=> specRelu1D 10 |=>
  specFullyConnected 10 1 |=> specLogit1D 1 |=> specNil1D 1
  where netSpecInner = specFullyConnected 40 30 |=> specRelu1D 30 |=> specFullyConnected 30 20 |=> specReshape1D2D 20 (2, 10) |=> specReshape2D1D (2, 10) 20 |=> specNil1D 20


-- | Specifications can be built using the following interface also. Here one does not have to carefully pay attention to match the dimension inputs and outputs as when using specification directly.
buildNetViaInterface :: IO SpecConcreteNetwork
buildNetViaInterface =
  buildModelWith (NetworkInitSettings UniformInit HMatrix) (DynamicBuildSetup { printResultingSpecification = False })  $
  inputLayer1D 2 >>                            -- 1. Every model has to start with an input dimension
  fullyConnected 10 >> dropout 0.89 >> relu >> -- 2. Layers are simply added as desired. The input of each layer is determined automatically,
  fullyConnected 20 >> relu >>                 --    whereas the output is specified as in `fullyConnected 20`. Thus, the layer empits 20 signals
  fullyConnected 4 >> tanhLayer >>             --    which arethe input of the next layer.
  networkLayer (
    inputLayer1D 4 >> fullyConnected 10 >> relu >> fullyConnected 4 >> sinusoid
    ) >>
  reshape (4, 1, 1) >>                         -- 3. Reshape is ignored if the input and output is the same
  fullyConnected 1                             -- 4. The output is simply the number of signals the alst layer emits.


netTrain ::
     (SingI (Last shapes), KnownNat len1, KnownNat len2, Head shapes ~ 'D1 len1, Last shapes ~ 'D1 len2)
  => Network layers shapes
  -> Optimizer o
  -> Int
  -> IO (Network layers shapes)
netTrain net0 op n = do
    -- inps <- replicateM n $ do
    --   s  <- getRandom
    --   return $ S1D $ SA.randomVector s SA.Uniform * 2 - 1
    -- let outs = flip map inps $ \(S1D v) ->
    --              if v `inCircle` (fromRational 0.50, 0.50)  || v `inCircle` (fromRational (-0.50), 0.50)
    --                then S1D $ fromRational 1
    --                else S1D $ fromRational 0
    gen <- createSystemRandom
    inps <-  replicateM n $ S1DV . V.map (\x->x*2-1) <$> (uniformVector gen 2)
    let outs = flip map inps $ \(S1DV v) ->
                 if v `inCircleV` (0.50, 0.50)  || v `inCircleV` (-0.50, 0.50)
                   then S1DV (V.singleton 1)
                   else S1DV (V.singleton 0)

    let trained = foldl' trainEach net0 (zip inps outs)
    return trained

  where trainEach !network (i,o) = train op network i o

netScore :: (KnownNat len, Head shapes ~ 'D1 len, Last shapes ~ 'D1 1) => Network layers shapes -> IO ()
netScore network = do
    let testIns = [ [ (x,y)  | x <- [0..50] ]
                             | y <- [0..20] ]
        -- outMat  = fmap (fmap (\(x,y) -> (render (x/25-1) (y/10-1) . normx) $ runNet network (S1D $ SA.vector [x / 25 - 1,y / 10 - 1]))) testIns
        outMat  = fmap (fmap (\(x,y) -> (render (x/25-1) (y/10-1) . normx) $ runNet network (S1DV $ V.fromList [x / 25 - 1,y / 10 - 1]))) testIns
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

testValues :: (KnownNat len, Head shapes ~ 'D1 len, Last shapes ~ 'D1 1) => Network layers shapes -> IO ()
testValues network = do
  -- inps <- replicateM 1000 $ do
  --     s  <- getRandom
  --     return $ S1D $ SA.randomVector s SA.Uniform * 2 - 1
  -- let outs = flip map inps $ \(S1D v) ->
  --                if v `inCircle` (fromRational 0.50, 0.50)  || v `inCircle` (fromRational (-0.50), 0.50)
  --                  then 1 :: Integer
  --                  else 0
  gen <- createSystemRandom
  inps <-  replicateM 1000 $ S1DV . V.map (\x->x*2-1) <$> uniformVector gen 2
  let outs = flip map inps $ \(S1DV v) ->
                 if v `inCircleV` (0.50, 0.50)  || v `inCircleV` (-0.50, 0.50)
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


inCircle :: KnownNat n => SA.R n -> (SA.R n, RealNum) -> Bool
v `inCircle` (o, r) = SA.norm_2 (v - o) <= r

inCircleV :: V.Vector RealNum -> (RealNum, RealNum) -> Bool
v `inCircleV` (o, r) = norm - o <= r
  where
    norm = sqrt (V.sum $ V.map (^ (2 :: Int)) v)


data FeedForwardOpts = FeedForwardOpts Int (Optimizer 'Adam)

feedForward' :: Parser FeedForwardOpts
feedForward' =
  FeedForwardOpts <$> option auto (long "examples" <> short 'e' <> value 1000)
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
       SpecConcreteNetwork1D1D (net0 :: Network layers shapes) <- networkFromSpecificationWith (NetworkInitSettings HeEtAl CBLAS) netSpec
      -- We need to specify the actual number of output nodes, as our functions requiere that!
       case (unsafeCoerce (Dict :: Dict ()) :: Dict (('D1 1) ~ Last shapes)) of
         Dict -> do
           net <- netTrain net0 rate examples
           unsafeCoerce $ -- only needed as GADTs are enabled, which disallowes the type to escape and thus prevents the type inference to work. The result is not needed anyways.
             testValues net)
    [1 .. nr]


  -- Features of dynamic networks:
  SpecConcreteNetwork1D1D (net' :: Network layers shapes) <- networkFromSpecificationWith (NetworkInitSettings HeEtAl CBLAS) netSpec
  net2 <- netTrain net' rate examples
  let spec' = networkToSpecification net2
  putStrLn "String represenation of the network specification: "
  print spec'
  let serializedSpec = encode spec'   -- only the specification (not the weights) are serialized here! The weights can be serialized using the networks serialize instance!
  let _ = encode net2                 -- E.g. like this.
  case decode serializedSpec of
    Left err -> print err
    Right spec2 -> do
      print spec2
      SpecConcreteNetwork1D1D (net3 :: Network layers3 shapes3) <- networkFromSpecificationWith (NetworkInitSettings HeEtAl CBLAS) spec2
      net4 <- foldM (\n _ -> netTrain n rate examples) net3 [(1 :: Int)..30]
      case (unsafeCoerce (Dict :: Dict ()) :: Dict (('D1 1) ~ Last shapes3)) of
        Dict -> netScore net4

  -- There is a also nice interface available also
  SpecConcreteNetwork1D1D newNet <- buildNetViaInterface
  print (networkToSpecification newNet)
