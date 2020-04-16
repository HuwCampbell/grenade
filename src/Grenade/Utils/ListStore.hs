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
    , getStep
    , listStoreToList
    , zipWithListStore
    ) where

import           Control.DeepSeq
import           Data.List       (foldl')
import           Data.Maybe
import           Data.Serialize
import           GHC.Generics

import           Grenade.Core

-- | Simple store
data ListStore a = ListStore Int [Maybe a]
  deriving (Generic, NFData, Serialize)

instance Functor ListStore where
  fmap f (ListStore n xs) = ListStore n $ map (fmap f) xs

-- | Returns how often the the store was updated, i.e. how often @setListStore@ was called (regardless of the parameters).
getStep :: ListStore a -> Int
getStep (ListStore n _) = n

listStoreToList :: ListStore a -> [Maybe a]
listStoreToList (ListStore _ xs) = xs

zipWithListStore :: (a -> b -> c) -> ListStore a -> ListStore b -> ListStore c
zipWithListStore f (ListStore n1 xs) (ListStore n2 ys) = ListStore (max n1 n2) $ zipWith (\x y -> f <$> x <*> y) xs ys

-- Elements are added when needed!
mkListStore :: ListStore a
mkListStore = ListStore 0 []

getListStore ::
     (LayerOptimizerData x (Optimizer o), MomentumStore x ~ ListStore (MomentumDataType x (Optimizer o)))
  => Optimizer o
  -> x
  -> MomentumStore x
  -> [MomentumDataType x (Optimizer o)]
getListStore opt x (ListStore _ xs)
  | maxIdx + 1 > length xs = getDataIndices (xs ++ replicate (maxIdx + 1 - length xs) Nothing)
  | otherwise = getDataIndices xs
  where
    indices = getIndices opt
    maxIdx = maximum indices
    getDataIndices xs' = fmap (\idx -> fromMaybe (newData opt x) (xs' !! idx)) indices


setListStore :: Optimizer o -> x -> ListStore (MomentumDataType x (Optimizer o)) -> [MomentumDataType x (Optimizer o)] -> ListStore (MomentumDataType x (Optimizer o))
setListStore opt _ (ListStore n xs) inp
  | length indices /= length inp = error $ "Wrong input length for " ++ show opt ++ " in setListStore in Grenade.Utils.ListStore: " ++ show (length inp)
  | otherwise = ListStore (n + 1) $ foldl' replace xs (zip [0 ..] indices)
  where
    indices = getIndices opt
    replace xs' (inIdx, idx) = take idx xs' ++ (Just (inp !! inIdx) : drop (idx + 1) xs')


getIndices :: Optimizer o -> [Int]
getIndices OptSGD{}  = [0]
getIndices OptAdam{} = [1,2]


instance (GNum a) => GNum (ListStore a) where
  s |* (ListStore n xs) = ListStore n $ map (fmap (s |*)) xs
  (ListStore n1 xs) |+ (ListStore n2 ys) = ListStore (max n1 n2) $ zipWith (\x y -> (|+) <$> x <*> y) xs ys
  gFromRational _ = mkListStore

