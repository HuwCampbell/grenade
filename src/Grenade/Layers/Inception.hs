{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-|
Module      : Grenade.Core.Network
Description : Core definition a simple neural etwork
Copyright   : (c) Huw Campbell, 2016-2017
License     : BSD2
Stability   : experimental
-}
module Grenade.Layers.Inception (
    Concat (..)
  ) where

import           Data.Singletons
import           GHC.TypeLits

import           Grenade.Core
import           Grenade.Layers.Convolution
import           Grenade.Layers.Pad
import           Grenade.Layers.Pooling
import           Grenade.Layers.Relu
import           Grenade.Layers.Concat

type Inception rows cols channels chx chy
  = Network '[ Concat ('D3 (rows - 5) (rows - 5) chx) (Inception3x3 rows cols channels chx) ('D3 (rows - 5) (rows - 5) chy) (Inception5x5 rows cols channels chy) ]
            '[ 'D3 rows cols channels, 'D3 (rows -2) (cols -2) (chx + chy) ]

type Inception3x3 rows cols channels chx
  = Network '[ Convolution channels chx 3 3 1 1 ]
            '[ 'D3 rows cols channels, 'D3 (rows -2) (cols -2) chx ]

type Inception5x5 rows cols channels chx
  = Network '[ Convolution channels chx 5 5 1 1, Pad 1 1 1 1 ]
            '[ 'D3 rows cols channels, 'D3 (rows -4) (cols -4) chx, 'D3 (rows -2) (cols -2) chx ]

