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
{-# LANGUAGE LambdaCase            #-}

module Grenade.Graph.Network (
    Layer (..)
  , UpdateLayer (..)
  ) where

import           Control.Monad.Random (MonadRandom)
import           Data.Singletons
import           Data.Singletons.Prelude

import           GHC.TypeLits

import           Grenade.Core.Shape
import           Grenade.Core.Network ( UpdateLayer (..), Layer (..) )

-- | Type of a DAG network

data Fin :: Nat -> * where
    Fin0 :: Fin (n + 1)
    FinS :: Fin n -> Fin (n + 1)

data Edge :: Nat -> * where
    Edge :: Shape -> Fin n -> Edge n

data Node a n where
    Node :: a -> [Edge n] -> Node a n
