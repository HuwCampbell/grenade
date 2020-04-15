{-# LANGUAGE AllowAmbiguousTypes   #-}
{-# LANGUAGE CPP                   #-}
{-# LANGUAGE ConstraintKinds       #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE DeriveAnyClass        #-}
{-# LANGUAGE DeriveGeneric         #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE UndecidableInstances  #-}
{-|
Module      : Grenade.Core.Layer
Description : Defines the Layer Classes
Copyright   : (c) Huw Campbell, 2016-2017
License     : BSD2
Stability   : experimental
-}

module Grenade.Utils.ListStore
    ( ListStore (..)
    , getListStore
    , setListStore
    , mkListStore
    ) where

import           Control.DeepSeq
import           Data.List       (foldl')
import           Data.Maybe
import           Data.Serialize
import           GHC.Generics

import           Grenade.Core

-- | Simple store
newtype ListStore a = ListStore [Maybe a]
  deriving (Generic, NFData, Serialize)

-- Elements are added when needed!
mkListStore :: ListStore a
mkListStore = ListStore []

getListStore ::
     (LayerOptimizerData x (Optimizer o), MomentumStore x ~ ListStore xData, MomentumData x (Optimizer o) ~ xData)
  => Optimizer o
  -> x
  -> MomentumStore x
  -> [MomentumData x (Optimizer o)]
getListStore opt x (ListStore xs)
  | maxIdx + 1 > length xs = getDataIndices (xs ++ replicate (maxIdx + 1 - length xs) Nothing)
  | otherwise = getDataIndices xs
  where
    indices = getIndices opt
    maxIdx = maximum indices
    getDataIndices xs' = fmap (\idx -> fromMaybe (newData opt x) (xs' !! idx)) indices


setListStore :: Optimizer o -> x -> ListStore (MomentumData x (Optimizer o)) -> [MomentumData x (Optimizer o)] -> ListStore (MomentumData x (Optimizer o))
setListStore opt _ (ListStore xs) inp
  | length indices /= length inp = error $ "Wrong input length for " ++ show opt ++ " in setListStore in Grenade.Utils.ListStore: " ++ show (length inp)
  | otherwise = ListStore $ foldl' replace xs (zip [0 ..] indices)
  where
    indices = getIndices opt
    replace xs' (inIdx, idx) = take idx xs' ++ (Just (inp !! inIdx) : drop (idx + 1) xs')


getIndices :: Optimizer o -> [Int]
getIndices OptSGD{} = [0]


instance (GNum a) => GNum (ListStore a) where
  s |* (ListStore xs) = ListStore $ map (fmap (s |*)) xs
  (ListStore xs) |+ (ListStore ys) = ListStore $ zipWith (\x y -> (|+) <$> x <*> y) xs ys
  gFromRational _ = error "Cannot call gFromRational for ListStore in Grenade.Utils.ListStore"
