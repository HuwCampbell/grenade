{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE InstanceSigs          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards       #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TypeOperators         #-}
module Grenade.Layers.BatchNorm (
    BatchNorm (..)
  ) where

import           Control.Monad.Random

import           Data.Proxy
import           Data.Serialize
import           Data.Singletons.TypeLits

import qualified Numeric.LinearAlgebra          as LA
import           Numeric.LinearAlgebra.Static   hiding (mean)

import           Grenade.Core

import           Debug.Trace
import           Grenade.Layers.Internal.Update

-- | A data type for holding mean and variance.
data BatchNorm i = BatchNorm Integer (R i) (R i) (R i) (R i) (R i)-- Counter, Mean and M2, variance, gamma, beta

-- data BatchGrads i = BatchGrads (BatchNorm i)

instance Show (BatchNorm i) where
  show BatchNorm {} = "BatchNorm"

instance (KnownNat i) => UpdateLayer (BatchNorm i) where
  type Gradient (BatchNorm i) = BatchNorm i

  runUpdate LearningParameters {..} _ batchNorm' = batchNorm'
  createRandom = return $ BatchNorm 0 0 0 0 1 0


eps :: (Fractional a) => a
eps = 0.0001

instance (KnownNat i) => Layer (BatchNorm i) ('D1 i) ('D1 i) where
  type Tape (BatchNorm i) ('D1 i) ('D1 i) = (R i,R i,R i,R i,R i)

  -- Do a matrix vector multiplication and return the result.
  runForwards (BatchNorm _ mean _ var ga be) (S1D v) =
    let -- xHat = (v-mean)* sqrt (stdDev+eps)
        y = ga*xHat+be
        xmu = v - mean
        stdDev = sqrt (var + eps)
        ivar = 1 / stdDev
        xHat = xmu * ivar
    in ((v,xmu,stdDev,ivar,xHat)
        --v
       , S1D y)

  -- Run a backpropogation step for a full connected layer.
  runBackwards (BatchNorm nr mean m2 var ga be) (v,xmu,stdDev,ivar,xHat) (S1D dEdy) =
    let (mean',m2',var') = onlineVariance nr mean m2 v
        -- be' = be + dEdy
        -- ga' = ga + (v - mean) * sqrt (var + eps) * dEdy

        -- dx = 1.0 / fromInteger nr * ga * sqrt (var + eps) * (fromInteger nr * dEdy - mean - (v - mean) * (var + eps)**(-1.0) * dEdy * (v - mean))


        be' = trace ("be'") be + dEdy
        ga' = ga + dEdy * xHat
        dxHat = dEdy * ga
        divar = divar + dxHat
        dxmu1 = trace ("dxmu1") dxHat * ivar
        dstdDev = -1 / (stdDev**2) * divar
        dvar = 0.5 / sqrt(var + eps) * dstdDev
        dsq = (1 / fromInteger nr) #> dvar
        dxmu2 = trace ("dxm2") 2 * xmu * dsq
        dx1 = dxmu1 + dxmu2
        dmu = -1 * (dxmu1+dxmu2)
        dx2 = trace ("dx2") 1 / fromInteger nr #> dmu
        dx = dx1 + dx2
    in (BatchNorm (nr+1) mean' m2' var' ga' be', S1D dx)


-- TODO Faster algorithm: http://cthorey.github.io./backpropagation/

-- # Forward pass
-- mu = 1/N*np.sum(h,axis =0) # Size (H,)
-- sigma2 = 1/N*np.sum((h-mu)**2,axis=0)# Size (H,)
-- hath = (h-mu)*(sigma2+epsilon)**(-1./2.)
-- y = gamma*hath+beta

-- Backward
--
-- mu = 1./N*np.sum(h, axis = 0)
-- var = 1./N*np.sum((h-mu)**2, axis = 0)
-- dbeta = np.sum(dy, axis=0)
-- dgamma = np.sum((h - mu) * (var + eps)**(-1. / 2.) * dy, axis=0)
-- dh = (1. / N) * gamma * (var + eps)**(-1. / 2.) * (N * dy - np.sum(dy, axis=0)
--     - (h - mu) * (var + eps)**(-1.0) * np.sum(dy * (h - mu), axis=0))

onlineVariance :: (Fractional a) => Integer -> a -> a -> a -> (a,a,a)
onlineVariance nrOfValues oldMean oldM2 value
  | n < 2 = (mean', m2, 1)
  | otherwise = (mean', m2, m2 / fromInteger nrOfValues)
  where n = nrOfValues + 1
        delta = value - oldMean
        mean' = oldMean + delta/fromInteger n
        delta2 = value - mean'
        m2 = oldM2 + delta*delta2

instance (KnownNat i) => Serialize (BatchNorm i) where
  put (BatchNorm nr mean m2 var ga be) = do
    put nr
    putListOf put . LA.toList . extract $ mean
    putListOf put . LA.toList . extract $ m2
    putListOf put . LA.toList . extract $ var
    putListOf put . LA.toList . extract $ ga
    putListOf put . LA.toList . extract $ be

  get = do
      nr <- get
      mean <- maybe (fail "Vector of incorrect size") return . create . LA.fromList =<< getListOf get
      m2 <- maybe (fail "Vector of incorrect size") return . create . LA.fromList =<< getListOf get
      var <- maybe (fail "Vector of incorrect size") return . create . LA.fromList =<< getListOf get
      ga <- maybe (fail "Vector of incorrect size") return . create . LA.fromList =<< getListOf get
      be <- maybe (fail "Vector of incorrect size") return . create . LA.fromList =<< getListOf get
      return $ BatchNorm nr mean m2 var ga be

-------------------- Num,Fractional,NMult instances --------------------

-- | Num and Fractional instance of Layer data type for calculating with networks
-- (slowly adapt target network, e.g. as in arXiv: 1509.02971)
instance (KnownNat i) => Num (BatchNorm i) where
  BatchNorm nr1 mean1 m12 var1 ga1 be1 + BatchNorm nr2 mean2 m22 var2 ga2 be2 = BatchNorm nr1 mean1 m12 var1 ga1 be1
  BatchNorm nr1 mean1 m12 var1 ga1 be1 * BatchNorm nr2 mean2 m22 var2 ga2 be2 = BatchNorm nr1 mean1 m12 var1 ga1 be1
  BatchNorm nr1 mean1 m12 var1 ga1 be1 - BatchNorm nr2 mean2 m22 var2 ga2 be2 = BatchNorm nr1 mean1 m12 var1 ga1 be1
  abs (BatchNorm nr mean m2 var ga be) = BatchNorm nr mean m2 var ga be
  signum (BatchNorm nr mean m2 var ga be) = BatchNorm nr mean m2 var ga be
  fromInteger _ = BatchNorm 0 0 0 0 1 0

instance (KnownNat i) => Fractional (BatchNorm i) where
  BatchNorm nr mean m2 var ga be / _ = BatchNorm nr mean m2 var ga be
  fromRational _ = BatchNorm 0 0 0 0 1 0


instance (KnownNat i) => NMult (BatchNorm i) where
  _ |* BatchNorm nr mean m2 var ga be = BatchNorm nr mean m2 var ga be

