{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE KindSignatures        #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE PolyKinds             #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE FlexibleInstances     #-}

-- Ghc 8.0 gives a warning on `(+)  _ _ = error ...` but ghc 7.10 fails to
-- compile without this default pattern.
{-# OPTIONS_GHC -fno-warn-overlapping-patterns #-}

module Grenade.Core.Shape (
    Shape (..)
  , S' (..)
  ) where

import           Data.Singletons.TypeLits
import           GHC.TypeLits

import           Numeric.LinearAlgebra.Static


-- | The current shapes we accept.
--   at the moment this is just one, two, and three dimensional
--   Vectors/Matricies.
data Shape =
    D1 Nat
  | D2 Nat Nat
  | D3 Nat Nat Nat

instance Num (S' x) where
  (+) (S1D' x) (S1D' y) = S1D' (x + y)
  (+) (S2D' x) (S2D' y) = S2D' (x + y)
  (+) (S3D' x) (S3D' y) = S3D' (x + y)
  (+)  _ _ = error "Impossible to have different constructors for the same shaped network"

  (-) (S1D' x) (S1D' y) = S1D' (x - y)
  (-) (S2D' x) (S2D' y) = S2D' (x - y)
  (-) (S3D' x) (S3D' y) = S3D' (x - y)
  (-)  _ _ = error "Impossible to have different constructors for the same shaped network"

  (*) (S1D' x) (S1D' y) = S1D' (x * y)
  (*) (S2D' x) (S2D' y) = S2D' (x * y)
  (*) (S3D' x) (S3D' y) = S3D' (x * y)
  (*)  _ _ = error "Impossible to have different constructors for the same shaped network"

  abs (S1D' x) = S1D' (abs x)
  abs (S2D' x) = S2D' (abs x)
  abs (S3D' x) = S3D' (abs x)

  signum (S1D' x) = S1D' (signum x)
  signum (S2D' x) = S2D' (signum x)
  signum (S3D' x) = S3D' (signum x)

  fromInteger _ = error "Unimplemented: fromInteger on Shape"

-- | Given a Shape n, these are the possible data structures with that shape.
--   All shapes are held in contiguous memory.
--   3D is held in a matrix (usually row oriented) which has height depth * rows.
data S' (n :: Shape) where
  S1D' :: ( KnownNat o )                      => R o             -> S' ('D1 o)
  S2D' :: ( KnownNat rows, KnownNat columns ) => L rows columns  -> S' ('D2 rows columns)
  S3D' :: ( KnownNat rows
          , KnownNat columns
          , KnownNat depth
          , KnownNat (rows * depth)) => L (rows * depth) columns -> S' ('D3 rows columns depth)

instance Show (S' n) where
  show (S1D' a) = "S1D' " ++ show a
  show (S2D' a) = "S2D' " ++ show a
  show (S3D' a) = "S3D' " ++ show a
