{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE RankNTypes            #-}

{-|
Module      : Grenade.Graph.Core.Network
Description : Core definition of the Shapes of data we understand
Copyright   : (c) Huw Campbell, 2016-2017
License     : BSD2
Stability   : experimental

This module defines the core data types for the shapes of data that
are understood by Grenade.
-}
module Grenade.Graph.Core.Network (
    Fin (..)
  ) where

import Data.Constraint
import Data.Singletons
import Data.Proxy

import Grenade

import GHC.TypeLits

data Edge :: * where
  E :: Fin n -> Shape -> Edge

data Fin n where
  Fin0 :: Fin (n + 1)
  FinS :: Fin n -> Fin (n + 1)

data Network :: [*] -> [Edge] -> * where { }

