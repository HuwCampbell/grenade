{-# LANGUAGE CPP                   #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE DeriveAnyClass        #-}
{-# LANGUAGE DeriveGeneric         #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE StandaloneDeriving    #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE UndecidableInstances  #-}
{-|
Module      : Grenade.Layers.Concat
Description : Concatenation layer
Copyright   : (c) Huw Campbell, 2016-2017
License     : BSD2
Stability   : experimental

This module provides the concatenation layer, which runs two chilld layers in parallel and combines their outputs.
-}
module Grenade.Layers.Concat
  ( Concat(..)
  -- , SpecConcat(..)
  ) where

import           Data.Serialize

import           Control.DeepSeq              (NFData (..))
import           Data.Singletons
import           GHC.Generics                 (Generic)
import           GHC.TypeLits

#if MIN_VERSION_base(4,9,0)
import           Data.Kind                    (Type)
#endif

import           Grenade.Core

import           Numeric.LinearAlgebra.Static (R, row, split, splitRows, unrow, ( # ),
                                               (===))

-- | A Concatenating Layer.
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
  deriving (Generic, NFData)

instance (Show x, Show y) => Show (Concat m x n y) where
  show (Concat x y) = "Concat\n" ++ show x ++ "\n" ++ show y

-- | Run two layers in parallel, combining their outputs.
instance (UpdateLayer x, UpdateLayer y) => UpdateLayer (Concat m x n y) where
  type Gradient (Concat m x n y) = (Gradient x, Gradient y)
  runUpdate lr (Concat x y) (x', y') = Concat (runUpdate lr x x') (runUpdate lr y y')

instance (RandomLayer x, RandomLayer y) => RandomLayer (Concat m x n y) where
  createRandomWith m gen = Concat <$> createRandomWith m gen <*> createRandomWith m gen

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


-------------------- DynamicNetwork instance --------------------


-- instance (FromDynamicLayer x, FromDynamicLayer y) => FromDynamicLayer (Concat m x n y) where
--   fromDynamicLayer inp (Concat x y) = SpecNetLayer $ SpecConcat (tripleFromSomeShape inp) (fromDynamicLayer x) (fromDynamicLayer y)


-- instance ToDynamicLayer SpecConcat where
--   toDynamicLayer wInit gen (SpecConcat xSpec ySpec) = do
--     x <- toDynamicLayer wInit gen xSpec
--     y <- toDynamicLayer wInit gen ySpec
--     return $ SpecLayer $ Concat x y


--   toDynamicLayer wInit gen (SpecFullyConnected nrI nrO) =
--     reifyNat nrI $ \(pxInp :: (KnownNat i') => Proxy i') ->
--       reifyNat nrO $ \(pxOut :: (KnownNat o') => Proxy o') ->
--         case (singByProxy pxInp %* singByProxy pxOut, unsafeCoerce (Dict :: Dict ()) :: Dict (i' ~ i), unsafeCoerce (Dict :: Dict ()) :: Dict (o' ~ o)) of
--           (SNat, Dict, Dict) -> do
--             (layer  :: FullyConnected i' o') <- randomFullyConnected wInit gen
--             return $ SpecLayer layer (SomeSing (sing :: Sing ('D1 i'))) (SomeSing (sing :: Sing ('D1 o')))


-- specFullyConnected :: Integer -> Integer -> SpecNet
-- specFullyConnected nrI nrO = SpecNetLayer $ SpecFullyConnected nrI nrO


-------------------- GNum instance --------------------


instance (GNum x, GNum y) => GNum (Concat m x n y) where
  n |* (Concat x y) = Concat (n |* x) (n |* y)
  (Concat x1 y1) |+ (Concat x2 y2) = Concat (x1 |+ x2) (y1 |+ y2)
  gFromRational r = Concat (gFromRational r) (gFromRational r)


