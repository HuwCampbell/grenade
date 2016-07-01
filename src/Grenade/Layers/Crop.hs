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

module Grenade.Layers.Crop (
    Crop (..)
  ) where

import           Data.Maybe
import           Data.Proxy
import           Data.Singletons.TypeLits
import           GHC.TypeLits

import           Grenade.Core.Network
import           Grenade.Core.Shape

import           Numeric.LinearAlgebra (konst, subMatrix, diagBlock)
import           Numeric.LinearAlgebra.Static (extract, create)

-- | A cropping layer for a neural network.
data Crop :: Nat
          -> Nat
          -> Nat
          -> Nat -> * where
  Crop :: ( KnownNat cropLeft
          , KnownNat cropTop
          , KnownNat cropRight
          , KnownNat cropBottom
          ) => Crop cropLeft cropTop cropRight cropBottom

instance Show (Crop cropLeft cropTop cropRight cropBottom) where
  show Crop = "Crop"

-- | A two dimentional image can be cropped.
instance ( Monad m
         , KnownNat cropLeft
         , KnownNat cropTop
         , KnownNat cropRight
         , KnownNat cropBottom
         , KnownNat inputRows
         , KnownNat inputColumns
         , KnownNat outputRows
         , KnownNat outputColumns
         , (inputRows - cropTop - cropBottom) ~ outputRows
         , (inputColumns - cropLeft - cropRight) ~ outputColumns
         ) => Layer m (Crop cropLeft cropTop cropRight cropBottom) ('D2 inputRows inputColumns) ('D2 outputRows outputColumns) where
  runForwards Crop (S2D' input) =
    let cropl = fromIntegral $ natVal (Proxy :: Proxy cropLeft)
        cropt = fromIntegral $ natVal (Proxy :: Proxy cropTop)
        nrows = fromIntegral $ natVal (Proxy :: Proxy outputRows)
        ncols = fromIntegral $ natVal (Proxy :: Proxy outputColumns)
        m  = extract input
        r  = subMatrix (cropt, cropl) (nrows, ncols) m
    in  return . S2D' . fromJust . create $ r
  runBackards _ crop _ (S2D' dEdy) =
    let cropl = fromIntegral $ natVal (Proxy :: Proxy cropLeft)
        cropt = fromIntegral $ natVal (Proxy :: Proxy cropTop)
        cropr = fromIntegral $ natVal (Proxy :: Proxy cropRight)
        cropb = fromIntegral $ natVal (Proxy :: Proxy cropBottom)
        eo    = extract dEdy
        vs    = diagBlock [konst 0 (cropt,cropl), eo, konst 0 (cropb,cropr)]
    in  return (crop, S2D' . fromJust . create $ vs)
