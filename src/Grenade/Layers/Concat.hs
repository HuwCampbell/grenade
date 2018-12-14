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
module Grenade.Layers.Concat (
    Concat (..)
  ) where

import           Data.Serialize

import           Data.Singletons
import           GHC.TypeLits

#if MIN_VERSION_base(4,9,0)
import           Data.Kind (Type)
#endif

import           Grenade.Core

import           Numeric.LinearAlgebra.Static ( row, (===), splitRows, unrow, (#), split, R )

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
data Concat :: Shape -> Type -> Shape -> Type -> Type where
  Concat :: x -> y -> Concat m x n y

instance (Show x, Show y) => Show (Concat m x n y) where
  show (Concat x y) = "Concat\n" ++ show x ++ "\n" ++ show y

-- | Run two layers in parallel, combining their outputs.
instance (UpdateLayer x, UpdateLayer y) => UpdateLayer (Concat m x n y) where
  type Gradient (Concat m x n y) = (Gradient x, Gradient y)
  runUpdate lr (Concat x y) (x', y') = Concat (runUpdate lr x x') (runUpdate lr y y')
  createRandom = Concat <$> createRandom <*> createRandom

instance ( SingI i
         , Layer x i ('D1 o)
         , Layer y i ('D1 o)
         ) => Layer (Concat ('D1 o) x ('D1 o) y) i ('D2 2 o) where
  type Tape (Concat ('D1 o) x ('D1 o) y) i ('D2 2 o) = (Tape x i ('D1 o), Tape y i ('D1 o))

  runForwards (Concat x y) input =
    let (xT, xOut :: S ('D1 o)) = runForwards x input
        (yT, yOut :: S ('D1 o)) = runForwards y input
    in case (xOut, yOut) of
        (S1D xOut', S1D yOut') ->
            ((xT, yT), S2D (row xOut' === row yOut'))

  runBackwards (Concat x y) (xTape, yTape) (S2D o) =
    let (ox, oy) = splitRows o
        (x', xB :: S i) = runBackwards x xTape (S1D $ unrow ox)
        (y', yB :: S i) = runBackwards y yTape (S1D $ unrow oy)
    in  ((x', y'), xB + yB)

instance ( SingI i
         , Layer x i ('D1 m)
         , Layer y i ('D1 n)
         , KnownNat o
         , KnownNat m
         , KnownNat n
         , o ~ (m + n)
         , n ~ (o - m)
         , (m <=? o) ~ 'True
         ) => Layer (Concat ('D1 m) x ('D1 n) y) i ('D1 o) where
  type Tape (Concat ('D1 m) x ('D1 n) y) i ('D1 o) = (Tape x i ('D1 m), Tape y i ('D1 n))

  runForwards (Concat x y) input =
    let (xT, xOut :: S ('D1 m)) = runForwards x input
        (yT, yOut :: S ('D1 n)) = runForwards y input
    in case (xOut, yOut) of
        (S1D xOut', S1D yOut') ->
            ((xT, yT), S1D (xOut' # yOut'))

  runBackwards (Concat x y) (xTape, yTape) (S1D o) =
    let (ox :: R m , oy :: R n) = split o
        (x', xB :: S i) = runBackwards x xTape (S1D ox)
        (y', yB :: S i) = runBackwards y yTape (S1D oy)
    in  ((x', y'), xB + yB)

-- | Concat 3D shapes, increasing the number of channels.
instance ( SingI i
         , Layer x i ('D3 rows cols m)
         , Layer y i ('D3 rows cols n)
         , KnownNat (rows * n)
         , KnownNat (rows * m)
         , KnownNat (rows * o)
         , KnownNat o
         , KnownNat m
         , KnownNat n
         , ((rows * m) + (rows * n)) ~ (rows * o)
         , ((rows * o) - (rows * m)) ~ (rows * n)
         , ((rows * m) <=? (rows * o)) ~ 'True
         ) => Layer (Concat ('D3 rows cols m) x ('D3 rows cols n) y) i ('D3 rows cols o) where
  type Tape (Concat ('D3 rows cols m) x ('D3 rows cols n) y) i ('D3 rows cols o) = (Tape x i ('D3 rows cols m), Tape y i ('D3 rows cols n))

  runForwards (Concat x y) input =
    let (xT, xOut :: S ('D3 rows cols m)) = runForwards x input
        (yT, yOut :: S ('D3 rows cols n)) = runForwards y input
    in case (xOut, yOut) of
        (S3D xOut', S3D yOut') ->
            ((xT, yT), S3D (xOut' === yOut'))

  runBackwards (Concat x y) (xTape, yTape) (S3D o) =
    let (ox, oy) = splitRows o
        (x', xB :: S i) = runBackwards x xTape (S3D ox :: S ('D3 rows cols m))
        (y', yB :: S i) = runBackwards y yTape (S3D oy :: S ('D3 rows cols n))
    in  ((x', y'), xB + yB)

instance (Serialize a, Serialize b) => Serialize (Concat sa a sb b) where
  put (Concat a b) = put a *> put b
  get = Concat <$> get <*> get
