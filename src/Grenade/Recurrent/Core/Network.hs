{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE EmptyDataDecls        #-}
module Grenade.Recurrent.Core.Network (
    Recurrent
  , FeedForward
  , RecurrentLayer (..)
  , RecurrentUpdateLayer (..)
  , RecurrentNetwork (..)
  , RecurrentInputs (..)
  , CreatableRecurrent (..)
  ) where


import           Control.Monad.Random ( MonadRandom )
import           Data.Singletons ( SingI )
import           Data.Serialize
import qualified Data.Vector.Storable as V

import           Grenade.Core

import qualified Numeric.LinearAlgebra as LA
import qualified Numeric.LinearAlgebra.Static as LAS


-- | Witness type to say indicate we're building up with a normal feed
--   forward layer.
data FeedForward :: * -> *
-- | Witness type to say indicate we're building up with a recurrent layer.
data Recurrent :: * -> *

-- | Class for a recurrent layer.
--   It's quite similar to a normal layer but for the input and output
--   of an extra recurrent data shape.
class UpdateLayer x => RecurrentUpdateLayer x where
  -- | Shape of data that is passed between each subsequent run of the layer
  type RecurrentShape x   :: Shape

class (RecurrentUpdateLayer x, SingI (RecurrentShape x)) => RecurrentLayer x (i :: Shape) (o :: Shape) where
  -- | Wengert Tape
  type RecTape x i o :: *
  -- | Used in training and scoring. Take the input from the previous
  --   layer, and give the output from this layer.
  runRecurrentForwards    :: x -> S (RecurrentShape x) -> S i -> (RecTape x i o, S (RecurrentShape x), S o)
  -- | Back propagate a step. Takes the current layer, the input that the
  --   layer gave from the input and the back propagated derivatives from
  --   the layer above.
  --   Returns the gradient layer and the derivatives to push back further.
  runRecurrentBackwards   :: x -> RecTape x i o -> S (RecurrentShape x) -> S o -> (Gradient x, S (RecurrentShape x), S i)

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

instance Show (RecurrentNetwork '[] '[i]) where
  show RNil = "NNil"
instance (Show x, Show (RecurrentNetwork xs rs)) => Show (RecurrentNetwork (FeedForward x ': xs) (i ': rs)) where
  show (x :~~> xs) = show x ++ "\n~~>\n" ++ show xs
instance (Show x, Show (RecurrentNetwork xs rs)) => Show (RecurrentNetwork (Recurrent x ': xs) (i ': rs)) where
  show (x :~@> xs) = show x ++ "\n~~>\n" ++ show xs

-- | Recurrent inputs (sideways shapes on an imaginary unrolled graph)
--   Parameterised on the layers of a Network.
data RecurrentInputs :: [*] -> * where
   RINil   :: RecurrentInputs '[]
   (:~~+>) :: UpdateLayer x
           => ()                   -> !(RecurrentInputs xs) -> RecurrentInputs (FeedForward x ': xs)
   (:~@+>) :: (SingI (RecurrentShape x), RecurrentUpdateLayer x)
           => !(S (RecurrentShape x)) -> !(RecurrentInputs xs) -> RecurrentInputs (Recurrent x ': xs)
infixr 5 :~~+>
infixr 5 :~@+>

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
