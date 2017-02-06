{-# LANGUAGE CPP                   #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE EmptyDataDecls        #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE ScopedTypeVariables   #-}

#if __GLASGOW_HASKELL__ < 800
{-# OPTIONS_GHC -fno-warn-incomplete-patterns #-}
#endif

module Grenade.Recurrent.Core.Network (
    Recurrent
  , FeedForward

  , RecurrentNetwork (..)
  , RecurrentInputs (..)
  , RecurrentTapes (..)
  , RecurrentGradients (..)

  , randomRecurrent
  , runRecurrentNetwork
  , runRecurrentGradient
  , applyRecurrentUpdate
  ) where


import           Control.Monad.Random ( MonadRandom )
import           Data.Singletons ( SingI )
import           Data.Singletons.Prelude ( Head, Last )
import           Data.Serialize
import qualified Data.Vector.Storable as V

import           Grenade.Core
import           Grenade.Recurrent.Core.Layer

import qualified Numeric.LinearAlgebra as LA
import qualified Numeric.LinearAlgebra.Static as LAS

-- | Witness type to say indicate we're building up with a normal feed
--   forward layer.
data FeedForward :: * -> *
-- | Witness type to say indicate we're building up with a recurrent layer.
data Recurrent :: * -> *

-- | Type of a recurrent neural network.
--
--   The [*] type specifies the types of the layers.
--
--   The [Shape] type specifies the shapes of data passed between the layers.
--
--   The definition is similar to a Network, but every layer in the
--   type is tagged by whether it's a FeedForward Layer of a Recurrent layer.
--
--   Often, to make the definitions more concise, one will use a type alias
--   for these empty data types.
data RecurrentNetwork :: [*] -> [Shape] -> * where
  RNil   :: SingI i
         => RecurrentNetwork '[] '[i]

  (:~~>) :: (SingI i, Layer x i h)
         => !x
         -> !(RecurrentNetwork xs (h ': hs))
         -> RecurrentNetwork (FeedForward x ': xs) (i ': h ': hs)

  (:~@>) :: (SingI i, RecurrentLayer x i h)
         => !x
         -> !(RecurrentNetwork xs (h ': hs))
         -> RecurrentNetwork (Recurrent x ': xs) (i ': h ': hs)
infixr 5 :~~>
infixr 5 :~@>

-- | Gradient of a network.
--
--   Parameterised on the layers of the network.
data RecurrentGradients :: [*] -> * where
   RGNil  :: RecurrentGradients '[]

   (://>) :: UpdateLayer x
          => [Gradient x]
          -> RecurrentGradients xs
          -> RecurrentGradients (phantom x ': xs)

-- | Recurrent inputs (sideways shapes on an imaginary unrolled graph)
--   Parameterised on the layers of a Network.
data RecurrentInputs :: [*] -> * where
   RINil   :: RecurrentInputs '[]

   (:~~+>) :: UpdateLayer x
           => ()                      -> !(RecurrentInputs xs) -> RecurrentInputs (FeedForward x ': xs)

   (:~@+>) :: (SingI (RecurrentShape x), RecurrentUpdateLayer x)
           => !(S (RecurrentShape x)) -> !(RecurrentInputs xs) -> RecurrentInputs (Recurrent x ': xs)

-- | All the information required to backpropogate
--   through time safely.
--
--   We index on the time step length as well, to ensure
--   that that all Tape lengths are the same.
data RecurrentTapes :: [*] -> [Shape] -> * where
   TRNil  :: SingI i
          => RecurrentTapes '[] '[i]

   (:\~>) :: [Tape x i h]
          -> !(RecurrentTapes xs (h ': hs))
          -> RecurrentTapes (FeedForward x ': xs) (i ': h ': hs)


   (:\@>) :: [RecTape x i h]
          -> !(RecurrentTapes xs (h ': hs))
          -> RecurrentTapes (Recurrent x ': xs) (i ': h ': hs)


runRecurrentNetwork  :: forall shapes layers.
                        RecurrentNetwork layers shapes
                     -> RecurrentInputs layers
                     -> [S (Head shapes)]
                     -> (RecurrentTapes layers shapes, RecurrentInputs layers, [S (Last shapes)])
runRecurrentNetwork =
  go
    where
  go  :: forall js sublayers. (Last js ~ Last shapes)
      => RecurrentNetwork sublayers js
      -> RecurrentInputs sublayers
      -> [S (Head js)]
      -> (RecurrentTapes sublayers js, RecurrentInputs sublayers, [S (Last js)])
  -- This is a simple non-recurrent layer, just map it forwards
  go (layer :~~> n) (() :~~+> nIn) !xs
      = let tys                 = runForwards layer <$> xs
            feedForwardTapes    = fst <$> tys
            forwards            = snd <$> tys
            -- recursively run the rest of the network, and get the gradients from above.
            (newFN, ig, answer) = go n nIn forwards
        in (feedForwardTapes :\~> newFN, () :~~+> ig, answer)

  -- This is a recurrent layer, so we need to do a scan, first input to last, providing
  -- the recurrent shape output to the next layer.
  go (layer :~@> n) (recIn :~@+> nIn) !xs
      = let (recOut, tys)       = goR layer recIn xs
            recurrentTapes      = fst <$> tys
            forwards            = snd <$> tys

            (newFN, ig, answer) = go n nIn forwards
        in (recurrentTapes :\@> newFN, recOut :~@+> ig, answer)

  -- Handle the output layer, bouncing the derivatives back down.
  -- We may not have a target for each example, so when we don't use 0 gradient.
  go RNil RINil !x
    = (TRNil, RINil, x)

  -- Helper function for recurrent layers
  -- Scans over the recurrent direction of the graph.
  goR !layer !recShape (x:xs) =
    let (tape, lerec, lepush) = runRecurrentForwards layer recShape x
        (rems, push)          = goR layer lerec xs
    in  (rems, (tape, lepush) : push)
  goR _ rin []      = (rin, [])

runRecurrentGradient :: forall layers shapes.
                        RecurrentNetwork layers shapes
                     -> RecurrentTapes layers shapes
                     -> RecurrentInputs layers
                     -> [S (Last shapes)]
                     -> (RecurrentGradients layers, RecurrentInputs layers, [S (Head shapes)])
runRecurrentGradient net tapes r o =
  go net tapes r
    where
  -- We have to be careful regarding the direction of the lists
  -- Inputs come in forwards, but our return value is backwards
  -- through time.
  go  :: forall js ss. (Last js ~ Last shapes)
      => RecurrentNetwork ss js
      -> RecurrentTapes ss js
      -> RecurrentInputs ss
      -> (RecurrentGradients ss, RecurrentInputs ss, [S (Head js)])
  -- This is a simple non-recurrent layer
  -- Run the rest of the network, then fmap the tapes and gradients
  go (layer :~~> n) (feedForwardTapes :\~> nTapes) (() :~~+> nRecs) =
    let (gradients, rins, feed)  = go n nTapes nRecs
        backs                    = uncurry (runBackwards layer) <$> zip (reverse feedForwardTapes) feed
    in  ((fst <$> backs) ://> gradients, () :~~+> rins, snd <$> backs)

  -- This is a recurrent layer
  -- Run the rest of the network, scan over the tapes in reverse
  go (layer :~@> n) (recurrentTapes :\@> nTapes) (recGrad :~@+> nRecs) =
    let (gradients, rins, feed)  = go n nTapes nRecs
        backExamples             = zip (reverse recurrentTapes) feed
        (rg, backs)              = goX layer recGrad backExamples
    in  ((fst <$> backs) ://> gradients, rg :~@+> rins, snd <$> backs)

  -- End of the road, so we reflect the given gradients backwards.
  -- Crucially, we reverse the list, so it's backwards in time as
  -- well.
  go RNil TRNil RINil
    = (RGNil, RINil, reverse o)

  -- Helper function for recurrent layers
  -- Scans over the recurrent direction of the graph.
  goX :: RecurrentLayer x i o => x -> S (RecurrentShape x) -> [(RecTape x i o, S o)] -> (S (RecurrentShape x), [(Gradient x, S i)])
  goX layer !lastback ((recTape, backgrad):xs) =
    let (layergrad, recgrad, ingrad) = runRecurrentBackwards layer recTape lastback backgrad
        (pushedback, ll)             = goX layer recgrad xs
    in  (pushedback, (layergrad, ingrad) : ll)
  goX _ !lastback []      = (lastback, [])

-- | Apply a batch of gradients to the network
--   Uses runUpdates which can be specialised for
--   a layer.
applyRecurrentUpdate :: LearningParameters
                     -> RecurrentNetwork layers shapes
                     -> RecurrentGradients layers
                     -> RecurrentNetwork layers shapes
applyRecurrentUpdate rate (layer :~~> rest) (gradient ://> grest)
  = runUpdates rate layer gradient :~~> applyRecurrentUpdate rate rest grest

applyRecurrentUpdate rate (layer :~@> rest) (gradient ://> grest)
  = runUpdates rate layer gradient :~@> applyRecurrentUpdate rate rest grest

applyRecurrentUpdate _ RNil RGNil
  = RNil


instance Show (RecurrentNetwork '[] '[i]) where
  show RNil = "NNil"
instance (Show x, Show (RecurrentNetwork xs rs)) => Show (RecurrentNetwork (FeedForward x ': xs) (i ': rs)) where
  show (x :~~> xs) = show x ++ "\n~~>\n" ++ show xs
instance (Show x, Show (RecurrentNetwork xs rs)) => Show (RecurrentNetwork (Recurrent x ': xs) (i ': rs)) where
  show (x :~@> xs) = show x ++ "\n~~>\n" ++ show xs


-- | A network can easily be created by hand with (:~~>) and (:~@>), but an easy way to initialise a random
--   recurrent network and a set of random inputs for it is with the randomRecurrent.
class CreatableRecurrent (xs :: [*]) (ss :: [Shape]) where
  -- | Create a network of the types requested
  randomRecurrent :: MonadRandom m => m (RecurrentNetwork xs ss, RecurrentInputs xs)

instance SingI i => CreatableRecurrent '[] '[i] where
  randomRecurrent =
    return (RNil, RINil)

instance (SingI i, Layer x i o, CreatableRecurrent xs (o ': rs)) => CreatableRecurrent (FeedForward x ': xs) (i ': o ': rs) where
  randomRecurrent = do
    thisLayer     <- createRandom
    (rest, resti) <- randomRecurrent
    return (thisLayer :~~> rest, () :~~+> resti)

instance (SingI i, RecurrentLayer x i o, CreatableRecurrent xs (o ':  rs)) => CreatableRecurrent (Recurrent x ': xs) (i ': o ': rs) where
  randomRecurrent = do
    thisLayer     <- createRandom
    thisShape     <- randomOfShape
    (rest, resti) <- randomRecurrent
    return (thisLayer :~@> rest, thisShape :~@+> resti)

-- | Add very simple serialisation to the recurrent network
instance SingI i => Serialize (RecurrentNetwork '[] '[i]) where
  put RNil = pure ()
  get = pure RNil

instance (SingI i, Layer x i o, Serialize x, Serialize (RecurrentNetwork xs (o ': rs))) => Serialize (RecurrentNetwork (FeedForward x ': xs) (i ': o ': rs)) where
  put (x :~~> r) = put x >> put r
  get = (:~~>) <$> get <*> get

instance (SingI i, RecurrentLayer x i o, Serialize x, Serialize (RecurrentNetwork xs (o ': rs))) => Serialize (RecurrentNetwork (Recurrent x ': xs) (i ': o ': rs)) where
  put (x :~@> r) = put x >> put r
  get = (:~@>) <$> get <*> get

instance (Serialize (RecurrentInputs '[])) where
  put _ = return ()
  get = return RINil

instance (UpdateLayer x, Serialize (RecurrentInputs ys)) => (Serialize (RecurrentInputs (FeedForward x ': ys))) where
  put ( () :~~+> rest) = put rest
  get = ( () :~~+> ) <$> get

instance (SingI (RecurrentShape x), RecurrentUpdateLayer x, Serialize (RecurrentInputs ys)) => (Serialize (RecurrentInputs (Recurrent x ': ys))) where
  put ( i :~@+> rest ) = do
    _ <- (case i of
           (S1D x) -> putListOf put . LA.toList . LAS.extract $ x
           (S2D x) -> putListOf put . LA.toList . LA.flatten . LAS.extract $ x
           (S3D x) -> putListOf put . LA.toList . LA.flatten . LAS.extract $ x
         ) :: PutM ()
    put rest

  get = do
    Just i <- fromStorable . V.fromList <$> getListOf get
    rest   <- get
    return ( i :~@+> rest)


-- Num instance for `RecurrentInputs layers`
-- Not sure if this is really needed, as I only need a `fromInteger 0` at
-- the moment for training, to create a null gradient on the recurrent
-- edge.
--
-- It does raise an interesting question though? Is a 0 gradient actually
-- the best?
--
-- I could imaging that weakly push back towards the optimum input could
-- help make a more stable generator.
instance (Num (RecurrentInputs '[])) where
  (+) _ _  = RINil
  (-) _ _  = RINil
  (*) _ _  = RINil
  abs _    = RINil
  signum _ = RINil
  fromInteger _ = RINil

instance (UpdateLayer x, Num (RecurrentInputs ys)) => (Num (RecurrentInputs (FeedForward x ': ys))) where
  (+) (() :~~+> x) (() :~~+> y)  = () :~~+> (x + y)
  (-) (() :~~+> x) (() :~~+> y)  = () :~~+> (x - y)
  (*) (() :~~+> x) (() :~~+> y)  = () :~~+> (x * y)
  abs (() :~~+> x)      = () :~~+> abs x
  signum (() :~~+> x)   = () :~~+> signum x
  fromInteger x         = () :~~+> fromInteger x

instance (SingI (RecurrentShape x), RecurrentUpdateLayer x, Num (RecurrentInputs ys)) => (Num (RecurrentInputs (Recurrent x ': ys))) where
  (+) (x :~@+> x') (y :~@+> y')  = (x + y) :~@+> (x' + y')
  (-) (x :~@+> x') (y :~@+> y')  = (x - y) :~@+> (x' - y')
  (*) (x :~@+> x') (y :~@+> y')  = (x * y) :~@+> (x' * y')
  abs (x :~@+> x')      = abs x :~@+> abs x'
  signum (x :~@+> x')   = signum x :~@+> signum x'
  fromInteger x         = fromInteger x :~@+> fromInteger x
