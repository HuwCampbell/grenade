{-# LANGUAGE CPP                   #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE UndecidableInstances  #-}
{-|
Module      : Grenade.Core.Pad
Description : Padding layer for 2D and 3D images
Copyright   : (c) Huw Campbell, 2016-2017
License     : BSD2
Stability   : experimental
-}
module Grenade.Layers.Pad (
    Pad (..)
  ) where

import           Data.Maybe
import           Data.Proxy
import           Data.Serialize
import           Data.Singletons.TypeLits

#if MIN_VERSION_base(4,11,0)
import           GHC.TypeLits hiding (natVal)
#else
import           GHC.TypeLits
#endif
#if MIN_VERSION_base(4,9,0)
import           Data.Kind (Type)
#endif

import           Grenade.Core
import           Grenade.Layers.Internal.Pad

import           Numeric.LinearAlgebra (konst, subMatrix, diagBlock)
import           Numeric.LinearAlgebra.Static (extract, create)

-- | A padding layer for a neural network.
--
--   Pads on the X and Y dimension of an image.
data Pad  :: Nat
          -> Nat
          -> Nat
          -> Nat -> Type where
  Pad  :: Pad padLeft padTop padRight padBottom

instance Show (Pad padLeft padTop padRight padBottom) where
  show Pad = "Pad"

instance UpdateLayer (Pad l t r b) where
  type Gradient (Pad l t r b) = ()
  runUpdate _ x _ = x
  createRandom = return Pad

instance Serialize (Pad l t r b) where
  put _ = return ()
  get = return Pad

-- | A two dimentional image can be padded.
instance ( KnownNat padLeft
         , KnownNat padTop
         , KnownNat padRight
         , KnownNat padBottom
         , KnownNat inputRows
         , KnownNat inputColumns
         , KnownNat outputRows
         , KnownNat outputColumns
         , (inputRows + padTop + padBottom) ~ outputRows
         , (inputColumns + padLeft + padRight) ~ outputColumns
         ) => Layer (Pad padLeft padTop padRight padBottom) ('D2 inputRows inputColumns) ('D2 outputRows outputColumns) where
  type Tape (Pad padLeft padTop padRight padBottom) ('D2 inputRows inputColumns) ('D2 outputRows outputColumns)  = ()
  runForwards Pad (S2D input) =
    let padl  = fromIntegral $ natVal (Proxy :: Proxy padLeft)
        padt  = fromIntegral $ natVal (Proxy :: Proxy padTop)
        padr  = fromIntegral $ natVal (Proxy :: Proxy padRight)
        padb  = fromIntegral $ natVal (Proxy :: Proxy padBottom)
        m     = extract input
        r     = diagBlock [konst 0 (padt,padl), m, konst 0 (padb,padr)]
    in  ((), S2D . fromJust . create $ r)
  runBackwards Pad _ (S2D dEdy) =
    let padl  = fromIntegral $ natVal (Proxy :: Proxy padLeft)
        padt  = fromIntegral $ natVal (Proxy :: Proxy padTop)
        nrows = fromIntegral $ natVal (Proxy :: Proxy inputRows)
        ncols = fromIntegral $ natVal (Proxy :: Proxy inputColumns)
        m     = extract dEdy
        vs    = subMatrix (padt, padl) (nrows, ncols) m
    in  ((), S2D . fromJust . create $ vs)

-- | A two dimentional image can be padded.
instance ( KnownNat padLeft
         , KnownNat padTop
         , KnownNat padRight
         , KnownNat padBottom
         , KnownNat inputRows
         , KnownNat inputColumns
         , KnownNat outputRows
         , KnownNat outputColumns
         , KnownNat channels
         , KnownNat (inputRows * channels)
         , KnownNat (outputRows * channels)
         , (inputRows + padTop + padBottom) ~ outputRows
         , (inputColumns + padLeft + padRight) ~ outputColumns
         ) => Layer (Pad padLeft padTop padRight padBottom) ('D3 inputRows inputColumns channels) ('D3 outputRows outputColumns channels) where
  type Tape (Pad padLeft padTop padRight padBottom) ('D3 inputRows inputColumns channels) ('D3 outputRows outputColumns channels)  = ()
  runForwards Pad (S3D input) =
    let padl  = fromIntegral $ natVal (Proxy :: Proxy padLeft)
        padt  = fromIntegral $ natVal (Proxy :: Proxy padTop)
        padr  = fromIntegral $ natVal (Proxy :: Proxy padRight)
        padb  = fromIntegral $ natVal (Proxy :: Proxy padBottom)
        outr  = fromIntegral $ natVal (Proxy :: Proxy outputRows)
        outc  = fromIntegral $ natVal (Proxy :: Proxy outputColumns)
        inr   = fromIntegral $ natVal (Proxy :: Proxy inputRows)
        inc   = fromIntegral $ natVal (Proxy :: Proxy inputColumns)
        ch    = fromIntegral $ natVal (Proxy :: Proxy channels)
        m     = extract input
        padded = pad ch padl padt padr padb inr inc outr outc m
    in  ((), S3D . fromJust . create $ padded)

  runBackwards Pad () (S3D gradient) =
    let padl  = fromIntegral $ natVal (Proxy :: Proxy padLeft)
        padt  = fromIntegral $ natVal (Proxy :: Proxy padTop)
        padr  = fromIntegral $ natVal (Proxy :: Proxy padRight)
        padb  = fromIntegral $ natVal (Proxy :: Proxy padBottom)
        outr  = fromIntegral $ natVal (Proxy :: Proxy outputRows)
        outc  = fromIntegral $ natVal (Proxy :: Proxy outputColumns)
        inr   = fromIntegral $ natVal (Proxy :: Proxy inputRows)
        inc   = fromIntegral $ natVal (Proxy :: Proxy inputColumns)
        ch    = fromIntegral $ natVal (Proxy :: Proxy channels)
        m     = extract gradient
        cropped = crop ch padl padt padr padb inr inc outr outc m
    in  ((), S3D . fromJust . create $ cropped)
