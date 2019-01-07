{-# LANGUAGE CPP                 #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE KindSignatures      #-}
{-# LANGUAGE NoStarIsType        #-}
{-# LANGUAGE PolyKinds           #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies        #-}
{-# LANGUAGE TypeOperators       #-}
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

import           Control.Monad
import           Data.Proxy
#if MIN_VERSION_base(4,11,0)
import           GHC.TypeLits                    hiding (natVal)
#else
import           GHC.TypeLits
#endif

<<<<<<< HEAD
=======
import           Data.Singletons.TypeLits hiding (natVal)
>>>>>>> natVal imports and no max versions in cabal file
import           GHC.TypeLits

import           Control.Monad.Primitive         (PrimBase, PrimState)
import           System.Random.MWC
import           System.Random.MWC.Distributions

import           Numeric.LinearAlgebra.Static


data WeightInitMethod = UniformInit       -- ^ W_l,i ~ U(-1/sqrt(n_l),1/sqrt(n_l))                   where n_l is the number of nodes in layer l
                      | Xavier            -- ^ W_l,i ~ U(-sqrt (6/n_l+n_{l+1}),sqrt (6/n_l+n_{l+1})) where n_l is the number of nodes in layer l
                      | HeEtAl            -- ^ W_l,i ~ N(0,sqrt(2/n_l))                              where n_l is the number of nodes in layer l


-- | Get a random vector initialized according to the specified method.
getRandomVector :: forall m n . (PrimBase m, KnownNat n) => Integer -> Integer -> WeightInitMethod -> Gen (PrimState m) -> m (R n)
getRandomVector i o method gen = do
  unifRands <- vector <$> replicateM n (uniformR (-1,1) gen)
  gaussRands <- vector <$> replicateM n (standard gen)

  return $ case method of
             UniformInit -> (1/sqrt (fromIntegral i)) * unifRands
             Xavier      -> (sqrt 6/sqrt (fromIntegral i + fromIntegral o)) * unifRands
             HeEtAl      -> sqrt (2/fromIntegral i) * gaussRands
  where n = fromIntegral $ natVal (Proxy :: Proxy n)


-- | Get a matrix with weights initialized according to the specified method.
getRandomMatrix :: forall m r n nr . (PrimBase m, KnownNat r, KnownNat n, KnownNat nr, nr ~ (n*r))
                => Integer -> Integer -> WeightInitMethod -> Gen (PrimState m) -> m (L r n)
getRandomMatrix i o method gen = do
  unifRands <- matrix <$> replicateM nr (uniformR (-1,1) gen)
  gaussRands <- matrix <$> replicateM nr (standard gen)

  return $ case method of
             UniformInit -> (1/sqrt (fromIntegral i)) * unifRands
             Xavier      -> (sqrt 6/sqrt (fromIntegral i + fromIntegral o)) * unifRands
             HeEtAl      -> (sqrt (2/fromIntegral i)) * gaussRands


  where nr = fromIntegral $ natVal (Proxy :: Proxy nr)
