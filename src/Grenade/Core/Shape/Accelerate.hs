{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE KindSignatures        #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE StandaloneDeriving    #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE RankNTypes            #-}
{-|
Module      : Grenade.Core.Shape
Description : Dependently typed shapes of data which are passed between layers of a network
Copyright   : (c) Huw Campbell, 2016-2017
License     : BSD2
Stability   : experimental


-}
module Grenade.Core.Shape.Accelerate
  ( S
  ) where

import           Data.Array.Accelerate

type S sh = Array sh Double
