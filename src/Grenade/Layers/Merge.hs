{-# LANGUAGE CPP                   #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE StandaloneDeriving    #-}
{-|
Module      : Grenade.Core.Network
Description : Merging layer for parallel network composition
Copyright   : (c) Huw Campbell, 2016-2017
License     : BSD2
Stability   : experimental
-}
module Grenade.Layers.Merge (
    Merge (..)
  ) where

import           Data.Serialize

import           Data.Singletons

#if MIN_VERSION_base(4,9,0)
import           Data.Kind (Type)
#endif

import           Grenade.Core

-- | A Merging layer.
--
-- Similar to Concat layer, except sums the activations instead of creating a larger
-- shape.
data Merge :: Type -> Type -> Type where
  Merge :: x -> y -> Merge x y

instance (Show x, Show y) => Show (Merge x y) where
  show (Merge x y) = "Merge\n" ++ show x ++ "\n" ++ show y

-- | Run two layers in parallel, combining their outputs.
--   This just kind of "smooshes" the weights together.
instance (UpdateLayer x, UpdateLayer y) => UpdateLayer (Merge x y) where
  type Gradient (Merge x y) = (Gradient x, Gradient y)
  runUpdate lr (Merge x y) (x', y') = Merge (runUpdate lr x x') (runUpdate lr y y')
  createRandom = Merge <$> createRandom <*> createRandom

-- | Combine the outputs and the inputs, summing the output shape
instance (SingI i, SingI o, Layer x i o, Layer y i o) => Layer (Merge x y) i o where
  type Tape (Merge x y) i o = (Tape x i o, Tape y i o)

  runForwards (Merge x y) input =
    let (xT, xOut) = runForwards x input
        (yT, yOut) = runForwards y input
    in  ((xT, yT), xOut + yOut)

  runBackwards (Merge x y) (xTape, yTape) o =
    let (x', xB) = runBackwards x xTape o
        (y', yB) = runBackwards y yTape o
    in  ((x', y'), xB + yB)

instance (Serialize a, Serialize b) => Serialize (Merge a b) where
  put (Merge a b) = put a *> put b
  get = Merge <$> get <*> get
