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
{-# LANGUAGE UndecidableInstances  #-}
{-|
Module      : Grenade.Layers.Concat
Description : Concatenation layer
Copyright   : (c) Huw Campbell, 2016-2017
License     : BSD2
Stability   : experimental

This module provides the concatenation layer, which runs two chilld layers in parallel and combines their outputs.
-}
module Grenade.Recurrent.Layers.ConcatRecurrent (
    ConcatRecurrent (..)
  ) where

import           Data.Serialize

import           Data.Singletons
import           GHC.TypeLits

#if MIN_VERSION_base(4,9,0)
import           Data.Kind (Type)
#endif

import           Grenade.Core
import           Grenade.Recurrent.Core

import           Numeric.LinearAlgebra.Static ( (#), split, R )

-- | A Concatentating Layer.
--
-- This layer shares it's input state between two sublayers, and concatenates their output.
--
-- With Networks able to be Layers, this allows for very expressive composition of complex Networks.
--
-- The Concat layer has a few instances, which allow one to flexibly "bash" together the outputs.
--
-- Two 1D vectors, can go to a 2D shape with 2 rows if their lengths are identical.
-- Any 2 1D vectors can also become a longer 1D Vector.
--
-- 3D images become 3D images with more channels. The sizes must be the same, one can use Pad
-- and Crop layers to ensure this is the case.
data ConcatRecurrent :: Shape -> Type -> Shape -> Type -> Type where
  ConcatRecLeft  :: x -> y -> ConcatRecurrent m (Recurrent x) n (FeedForward y)
  ConcatRecRight :: x -> y -> ConcatRecurrent m (FeedForward x) n (Recurrent y)
  ConcatRecBoth  :: x -> y -> ConcatRecurrent m (Recurrent x) n   (Recurrent y)

instance (Show x, Show y) => Show (ConcatRecurrent m (p x) n (q y)) where
  show (ConcatRecLeft x y)  = "ConcatRecLeft\n" ++ show x ++ "\n" ++ show y
  show (ConcatRecRight x y) = "ConcatRecRight\n" ++ show x ++ "\n" ++ show y
  show (ConcatRecBoth x y)  = "ConcatRecBoth\n" ++ show x ++ "\n" ++ show y

instance (RecurrentUpdateLayer x, UpdateLayer y) => UpdateLayer (ConcatRecurrent m (Recurrent x) n (FeedForward y)) where
  type Gradient (ConcatRecurrent m (Recurrent x) n (FeedForward y)) = (Gradient x, Gradient y)
  runUpdate lr (ConcatRecLeft x y) (x', y')
    = ConcatRecLeft (runUpdate lr x x') (runUpdate lr y y')
  createRandom
    = ConcatRecLeft <$> createRandom <*> createRandom

instance (UpdateLayer x, RecurrentUpdateLayer y) => UpdateLayer (ConcatRecurrent m (FeedForward x) n (Recurrent y)) where
  type Gradient (ConcatRecurrent m (FeedForward x) n (Recurrent y)) = (Gradient x, Gradient y)
  runUpdate lr (ConcatRecRight x y) (x', y')
    = ConcatRecRight (runUpdate lr x x') (runUpdate lr y y')
  createRandom
    = ConcatRecRight <$> createRandom <*> createRandom

instance (RecurrentUpdateLayer x, RecurrentUpdateLayer y) => UpdateLayer (ConcatRecurrent m (Recurrent x) n (Recurrent y)) where
  type Gradient (ConcatRecurrent m (Recurrent x) n (Recurrent y)) = (Gradient x, Gradient y)
  runUpdate lr (ConcatRecBoth x y) (x', y')
    = ConcatRecBoth (runUpdate lr x x') (runUpdate lr y y')
  createRandom
    = ConcatRecBoth <$> createRandom <*> createRandom

instance (RecurrentUpdateLayer x, UpdateLayer y) => RecurrentUpdateLayer (ConcatRecurrent m (Recurrent x) n (FeedForward y)) where
  type RecurrentShape (ConcatRecurrent m (Recurrent x) n (FeedForward y)) = RecurrentShape x

instance (UpdateLayer x, RecurrentUpdateLayer y) => RecurrentUpdateLayer (ConcatRecurrent m (FeedForward x) n (Recurrent y)) where
  type RecurrentShape (ConcatRecurrent m (FeedForward x) n (Recurrent y)) = RecurrentShape y

instance (RecurrentUpdateLayer x, RecurrentUpdateLayer y) => RecurrentUpdateLayer (ConcatRecurrent m (Recurrent x) n (Recurrent y)) where
  type RecurrentShape (ConcatRecurrent m (Recurrent x) n (Recurrent y)) =  RecurrentInputs '[ Recurrent x, Recurrent y ]

instance ( SingI i
         , Layer x i ('D1 m)
         , RecurrentLayer y i ('D1 n)
         , KnownNat o
         , KnownNat m
         , KnownNat n
         , o ~ (m + n)
         , n ~ (o - m)
         , (m <=? o) ~ 'True
         ) => RecurrentLayer (ConcatRecurrent ('D1 m) (FeedForward x) ('D1 n) (Recurrent y)) i ('D1 o) where
  type RecTape (ConcatRecurrent ('D1 m) (FeedForward x) ('D1 n) (Recurrent y)) i ('D1 o) = (Tape x i ('D1 m), RecTape y i ('D1 n))

  runRecurrentForwards (ConcatRecRight x y) s input =
    let (xT, xOut :: S ('D1 m))       = runForwards x input
        (yT, side, yOut :: S ('D1 n)) = runRecurrentForwards y s input
    in case (xOut, yOut) of
        (S1D xOut', S1D yOut') ->
            ((xT, yT), side, S1D (xOut' # yOut'))

  runRecurrentBackwards (ConcatRecRight x y) (xTape, yTape) s (S1D o) =
    let (ox :: R m , oy :: R n) = split o
        (x', xB :: S i)         = runBackwards x xTape (S1D ox)
        (y', side, yB :: S i)   = runRecurrentBackwards y yTape s (S1D oy)
    in  ((x', y'), side, xB + yB)

instance ( SingI i
         , RecurrentLayer x i ('D1 m)
         , Layer y i ('D1 n)
         , KnownNat o
         , KnownNat m
         , KnownNat n
         , o ~ (m + n)
         , n ~ (o - m)
         , (m <=? o) ~ 'True
         ) => RecurrentLayer (ConcatRecurrent ('D1 m) (Recurrent x) ('D1 n) (FeedForward y)) i ('D1 o) where
  type RecTape (ConcatRecurrent ('D1 m) (Recurrent x) ('D1 n) (FeedForward y)) i ('D1 o) = (RecTape x i ('D1 m), Tape y i ('D1 n))

  runRecurrentForwards (ConcatRecLeft x y) s input =
    let (xT, side, xOut :: S ('D1 m)) = runRecurrentForwards x s input
        (yT, yOut :: S ('D1 n))       = runForwards y input
    in case (xOut, yOut) of
        (S1D xOut', S1D yOut') ->
            ((xT, yT), side, S1D (xOut' # yOut'))

  runRecurrentBackwards (ConcatRecLeft x y) (xTape, yTape) s (S1D o) =
    let (ox :: R m , oy :: R n) = split o
        (x', side, xB :: S i)   = runRecurrentBackwards x xTape s (S1D ox)
        (y', yB :: S i)         = runBackwards y yTape (S1D oy)
    in  ((x', y'), side, xB + yB)

instance ( SingI i
         , RecurrentLayer x i ('D1 m)
         , RecurrentLayer y i ('D1 n)
         , Fractional (RecurrentShape x)
         , Fractional (RecurrentShape y)
         , KnownNat o
         , KnownNat m
         , KnownNat n
         , o ~ (m + n)
         , n ~ (o - m)
         , (m <=? o) ~ 'True
         ) => RecurrentLayer (ConcatRecurrent ('D1 m) (Recurrent x) ('D1 n) (Recurrent y)) i ('D1 o) where
  type RecTape (ConcatRecurrent ('D1 m) (Recurrent x) ('D1 n) (Recurrent y)) i ('D1 o) = (RecTape x i ('D1 m), RecTape y i ('D1 n))

  runRecurrentForwards (ConcatRecBoth x y) (sx :~@+> (sy :~@+> RINil)) input =
    let (xT, s'x, xOut :: S ('D1 m)) = runRecurrentForwards x sx input
        (yT, s'y, yOut :: S ('D1 n)) = runRecurrentForwards y sy input
    in case (xOut, yOut) of
        (S1D xOut', S1D yOut') ->
            ((xT, yT), (s'x :~@+> (s'y :~@+> RINil)), S1D (xOut' # yOut'))

  runRecurrentBackwards (ConcatRecBoth x y) (xTape, yTape) (sx :~@+> (sy :~@+> RINil)) (S1D o) =
    let (ox :: R m , oy :: R n) = split o
        (x', s'x, xB :: S i)    = runRecurrentBackwards x xTape sx (S1D ox)
        (y', s'y, yB :: S i)    = runRecurrentBackwards y yTape sy (S1D oy)
    in  ((x', y'), (s'x :~@+> (s'y :~@+> RINil)), xB + yB)

instance (Serialize a, Serialize b) => Serialize (ConcatRecurrent sa (FeedForward a) sb (Recurrent b)) where
  put (ConcatRecRight a b) = put a *> put b
  get = ConcatRecRight <$> get <*> get

instance (Serialize a, Serialize b) => Serialize (ConcatRecurrent sa (Recurrent a) sb (Recurrent b)) where
  put (ConcatRecBoth a b) = put a *> put b
  get = ConcatRecBoth <$> get <*> get

instance (Serialize a, Serialize b) => Serialize (ConcatRecurrent sa (Recurrent a) sb (FeedForward b)) where
  put (ConcatRecLeft a b) = put a *> put b
  get = ConcatRecLeft <$> get <*> get
