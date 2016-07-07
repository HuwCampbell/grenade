{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE StandaloneDeriving    #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE PolyKinds             #-}

module Grenade.Layers.Pad (
    Pad (..)
  ) where

import           Data.Maybe
import           Data.Proxy
import           Data.Singletons.TypeLits
import           GHC.TypeLits

import           Grenade.Core.Network
import           Grenade.Core.Shape

import           Numeric.LinearAlgebra (konst, subMatrix, diagBlock)
import           Numeric.LinearAlgebra.Static (extract, create)

-- | A padding layer for a neural network.
data Pad  :: Nat
          -> Nat
          -> Nat
          -> Nat -> * where
  Pad  :: ( KnownNat padLeft
          , KnownNat padTop
          , KnownNat padRight
          , KnownNat padBottom
          ) => Pad padLeft padTop padRight padBottom

instance Show (Pad padLeft padTop padRight padBottom) where
  show Pad = "Pad"

-- | A two dimentional image can be padped.
instance ( Monad m
         , KnownNat padLeft
         , KnownNat padTop
         , KnownNat padRight
         , KnownNat padBottom
         , KnownNat inputRows
         , KnownNat inputColumns
         , KnownNat outputRows
         , KnownNat outputColumns
         , (inputRows + padTop + padBottom) ~ outputRows
         , (inputColumns + padLeft + padRight) ~ outputColumns
         ) => Layer m (Pad padLeft padTop padRight padBottom) ('D2 inputRows inputColumns) ('D2 outputRows outputColumns) where
  runForwards Pad (S2D' input) =
    let padl  = fromIntegral $ natVal (Proxy :: Proxy padLeft)
        padt  = fromIntegral $ natVal (Proxy :: Proxy padTop)
        padr  = fromIntegral $ natVal (Proxy :: Proxy padRight)
        padb  = fromIntegral $ natVal (Proxy :: Proxy padBottom)
        m     = extract input
        r     = diagBlock [konst 0 (padt,padl), m, konst 0 (padb,padr)]
    in  return . S2D' . fromJust . create $ r
  runBackards _ pad _ (S2D' dEdy) =
    let padl  = fromIntegral $ natVal (Proxy :: Proxy padLeft)
        padt  = fromIntegral $ natVal (Proxy :: Proxy padTop)
        nrows = fromIntegral $ natVal (Proxy :: Proxy inputRows)
        ncols = fromIntegral $ natVal (Proxy :: Proxy inputColumns)
        m     = extract dEdy
        vs    = subMatrix (padt, padl) (nrows, ncols) m
    in  return (pad, S2D' . fromJust . create $ vs)
