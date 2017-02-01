{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE FlexibleInstances     #-}
module Grenade.Layers.Fuse (
    SubNetwork (..)
  , BranchNetwork (..)
  ) where

import           Data.Singletons.Prelude

import           Grenade.Core.Network
import           Grenade.Core.Shape
import           Grenade.Core.Runner

-- | Fuse two layers into one layer.
--   This can be used to simplify a network if a complicated repeated structure is used.
data SubNetwork :: [*] -> [Shape] -> * where
     SubNetwork :: Network sublayers subshapes -> SubNetwork sublayers subshapes

instance CreatableNetwork sublayers subshapes => UpdateLayer (SubNetwork sublayers subshapes) where
  type Gradient (SubNetwork sublayers subshapes) = Gradients sublayers
  runUpdate lr (SubNetwork net) = SubNetwork . applyUpdate lr net
  createRandom = SubNetwork <$> randomNetwork

instance (CreatableNetwork sublayers subshapes, i ~ (Head subshapes), o ~ (Last subshapes)) => Layer (SubNetwork sublayers subshapes) i o where
  type Tape (SubNetwork sublayers subshapes) i o = Tapes sublayers subshapes

  runForwards (SubNetwork net) i =
    go i net
      where
    go  :: forall js ss. (Last js ~ Last subshapes)
        => S (Head js)          -- ^ input vector
        -> Network ss js -- ^ network to train
        -> (Tapes ss js, S (Last js))
    go !x (layer :~> n) =
      let (tape, forward) = runForwards layer x
          (tapes, answer) = go forward n
      in  (tape :\> tapes, answer)
    go !x NNil
        = (TNil, x)

  runBackwards (SubNetwork net) tapes o =
    go net tapes
      where
    go  :: forall js ss. (Last js ~ Last subshapes)
        => Network ss js -- ^ network to train
        -> Tapes ss js -- ^ network to train
        -> (Gradients ss, S (Head js))
    go (layer :~> n) (tape :\> nt) =
      let (gradients, feed)  = go n nt
          (layer', backGrad) = runBackwards layer tape feed
      in  (layer' :/> gradients, backGrad)
    go NNil TNil
        = (GNil, o)


-- | Run two layers in parallel, combining their outputs. The way in which the output is combined is dependent
--   on the inut
data BranchNetwork :: * -> * -> * where
     BranchNetwork :: x -> y -> BranchNetwork x y

instance (UpdateLayer x, UpdateLayer y) => UpdateLayer (BranchNetwork x y) where
  type Gradient (BranchNetwork x y) = (Gradient x, Gradient y)
  runUpdate lr (BranchNetwork x y) (x', y') = BranchNetwork (runUpdate lr x x') (runUpdate lr y y')
  createRandom = BranchNetwork <$> createRandom <*> createRandom

instance (SingI i, SingI o, Layer x i o, Layer y i o) => Layer (BranchNetwork x y) i o where
  type Tape (BranchNetwork x y) i o  = (Tape x i o, Tape y i o)
  runForwards (BranchNetwork x y) input =
    let (xT, xOut) = runForwards x input
        (yT, yOut) = runForwards y input
    in  ((xT, yT), xOut + yOut)

  runBackwards (BranchNetwork x y) (xTape, yTape) o =
    let (x', xB) = runBackwards x xTape o
        (y', yB) = runBackwards y yTape o
    in  ((x', y'), xB + yB)
