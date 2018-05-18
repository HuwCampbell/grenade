{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}

module Grenade.Run
    ( run
    , DataSets(..)
    ) where

import Grenade.Train.DataSet
import Grenade.Train.Network

import Data.Singletons (SingI)
import Data.Singletons.Prelude (Head, Last)

import Grenade.Core (LearningParameters, Network, Shape)

import Control.Monad

run :: forall (shapes :: [Shape]) (layers :: [*]) (i :: Shape) (o :: Shape).
       (SingI o, i ~ Head shapes, o ~ Last shapes)
    => Int
    -> LearningParameters
    -> IO (Network layers shapes)
    -> IO (DataSets i o)
    -> IO ()
run epochs param networkM loadData = do
    DataSets trainSet valSet testSet <- loadData
    net0 <- networkM
    void $
        trainNetworkAndPrintAccuracies epochs param trainSet valSet testSet net0

data DataSets (i :: Shape) (o :: Shape) = DataSets
    { trainDataSet :: DataSet i o
    , valDataSet :: DataSet i o
    , testDataSet :: DataSet i o
    } deriving (Show)
