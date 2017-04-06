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
Description : Inception style parallel convolutional network composition.
Copyright   : (c) Huw Campbell, 2016-2017
License     : BSD2
Stability   : experimental

Export an Inception style type, which can be used to build up
complex multiconvolution size networks.
-}
module Grenade.Layers.Inception (
    Inception
  , InceptionMini
  , Resnet
  ) where

import           GHC.TypeLits

import           Grenade.Core
import           Grenade.Layers.Convolution
import           Grenade.Layers.Pad
import           Grenade.Layers.Concat
import           Grenade.Layers.Merge
import           Grenade.Layers.Trivial

-- | Type of an inception layer.
--
--   It looks like a bit of a handful, but is actually pretty easy to use.
--
--   The first three type parameters are the size of the (3D) data the
--   inception layer will take. It will emit 3D data with the number of
--   channels being the sum of @chx@, @chy@, @chz@, which are the number
--   of convolution filters in the 3x3, 5x5, and 7x7 convolutions Layers
--   respectively.
--
--   The network get padded effectively before each convolution filters
--   such that the output dimension is the same x and y as the input.
type Inception rows cols channels chx chy chz
  = Network '[ Concat ('D3 rows cols (chx + chy)) (InceptionMini rows cols channels chx chy) ('D3 rows cols chz) (Inception7x7 rows cols channels chz) ]
            '[ 'D3 rows cols channels, 'D3 rows cols (chx + chy + chz) ]

type InceptionMini rows cols channels chx chy
  = Network '[ Concat ('D3 rows cols chx) (Inception3x3 rows cols channels chx) ('D3 rows cols chy) (Inception5x5 rows cols channels chy) ]
            '[ 'D3 rows cols channels, 'D3 rows cols (chx + chy) ]

type Inception3x3 rows cols channels chx
  = Network '[ Pad 1 1 1 1, Convolution channels chx 3 3 1 1 ]
            '[ 'D3 rows cols channels, 'D3 (rows + 2) (cols + 2) channels, 'D3 rows cols chx ]

type Inception5x5 rows cols channels chx
  = Network '[ Pad 2 2 2 2, Convolution channels chx 5 5 1 1 ]
            '[ 'D3 rows cols channels, 'D3 (rows + 4) (cols + 4) channels, 'D3 rows cols chx ]

type Inception7x7 rows cols channels chx
  = Network '[ Pad 3 3 3 3, Convolution channels chx 7 7 1 1 ]
            '[ 'D3 rows cols channels, 'D3 (rows + 6) (cols + 6) channels, 'D3 rows cols chx ]

type Resnet branch = Merge Trivial branch
