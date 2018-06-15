{-# LANGUAGE ConstraintKinds       #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TypeOperators         #-}
{-|
Module      : Grenade.Core.WeightInitialization
Description : Defines the Weight Initialization methods of Grenade.
Copyright   : (c) Manuel Schneckenreither, 2018
License     : BSD2
Stability   : experimental

This module defines the weight initialization methods.

-}


module Grenade.Core.WeightInitialization
    ( getRandomVector
    , getRandomMatrix
    , WeightInitMethod (..)
    ) where

import           Control.Monad.Random

import           Data.Singletons.TypeLits
import           Numeric.LinearAlgebra.Static


data WeightInitMethod = UniformInit       -- ^ W_l,i ~ U(-1/sqrt(n_l),1/sqrt(n_l))                   where n_l is the number of nodes in layer l
                      | Xavier            -- ^ W_l,i ~ U(-sqrt (6/n_l+n_{l+1}),sqrt (6/n_l+n_{l+1})) where n_l is the number of nodes in layer l
                      | HeEtAl            -- ^ W_l,i ~ N(0,2/n_l)                                    where n_l is the number of nodes in layer l


-- | Get a random vector initialized according to the specified method.
getRandomVector :: (MonadRandom m, KnownNat n) => Integer -> Integer -> WeightInitMethod -> m (R n)
getRandomVector i o method = do
  s <- getRandom
  return $ case method of
             UniformInit -> (1/sqrt (fromIntegral i)) * (randomVector s Uniform * 2 - 1)
             Xavier -> (sqrt 6/sqrt (fromIntegral i + fromIntegral o)) * (randomVector s Uniform * 2 - 1)
             HeEtAl -> sqrt (2/fromIntegral i) * randomVector s Gaussian

-- | Get a matrix with weights initialized according to the specified method.
getRandomMatrix :: (MonadRandom m, KnownNat r, KnownNat n) => Integer -> Integer -> WeightInitMethod -> m (L r n)
getRandomMatrix i o method = do
  s <- getRandom
  return $ case method of
             UniformInit -> (1/sqrt (fromIntegral i)) * uniformSample s (-1) 1
             Xavier -> (sqrt 6/sqrt (fromIntegral i + fromIntegral o)) * uniformSample s (-1) 1
             HeEtAl -> gaussianSample s 0 (sym $ diag $ sqrt (2/fromIntegral i))


