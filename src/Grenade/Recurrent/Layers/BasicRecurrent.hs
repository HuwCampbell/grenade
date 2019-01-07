{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE InstanceSigs          #-}
{-# LANGUAGE KindSignatures        #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PolyKinds             #-}
{-# LANGUAGE RecordWildCards       #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE UndecidableInstances  #-}

module Grenade.Recurrent.Layers.BasicRecurrent (
    BasicRecurrent (..)
  ) where


import           Data.Proxy
import           Data.Singletons.TypeLits hiding (natVal)

import           Numeric.LinearAlgebra.Static

import           GHC.TypeLits

import           Grenade.Core
import           Grenade.Recurrent.Core

data BasicRecurrent :: Nat -- Input layer size
                    -> Nat -- Output layer size
                    -> * where
  BasicRecurrent :: ( KnownNat input
                    , KnownNat output
                    , KnownNat matrixCols
                    , matrixCols ~ (input + output))
                 => !(R output)   -- Bias neuron weights
                 -> !(R output)   -- Bias neuron momentum
                 -> !(L output matrixCols) -- Activation
                 -> !(L output matrixCols) -- Momentum
                 -> BasicRecurrent input output

data BasicRecurrent' :: Nat -- Input layer size
                     -> Nat -- Output layer size
                     -> * where
  BasicRecurrent' :: ( KnownNat input
                     , KnownNat output
                     , KnownNat matrixCols
                     , matrixCols ~ (input + output))
                  => !(R output)   -- Bias neuron gradients
                  -> !(L output matrixCols)
                  -> BasicRecurrent' input output

instance Show (BasicRecurrent i o) where
  show BasicRecurrent {} = "BasicRecurrent"

instance (KnownNat i, KnownNat o, KnownNat (i + o)) => UpdateLayer (BasicRecurrent i o) where
  type Gradient (BasicRecurrent i o) = (BasicRecurrent' i o)

  runUpdate LearningParameters {..} (BasicRecurrent oldBias oldBiasMomentum oldActivations oldMomentum) (BasicRecurrent' biasGradient activationGradient) =
    let newBiasMomentum = konst learningMomentum * oldBiasMomentum - konst learningRate * biasGradient
        newBias         = oldBias + newBiasMomentum
        newMomentum     = konst learningMomentum * oldMomentum - konst learningRate * activationGradient
        regulariser     = konst (learningRegulariser * learningRate) * oldActivations
        newActivations  = oldActivations + newMomentum - regulariser
    in BasicRecurrent newBias newBiasMomentum newActivations newMomentum

instance (KnownNat i, KnownNat o, KnownNat x, KnownNat (x*o), x ~ (i+o)) => RandomLayer (BasicRecurrent i o) where
  createRandomWith m gen = do
    wB <- getRandomVector i o m gen
    wN <- getRandomMatrix i o m gen
    let bm = konst 0
        mm = konst 0
    return $ BasicRecurrent wB bm wN mm
      where i = natVal (Proxy :: Proxy i)
            o = natVal (Proxy :: Proxy o)


instance (KnownNat i, KnownNat o, KnownNat (i + o), i <= (i + o), o ~ ((i + o) - i)) => RecurrentUpdateLayer (BasicRecurrent i o) where
  type RecurrentShape (BasicRecurrent i o) = S ('D1 o)

instance (KnownNat i, KnownNat o, KnownNat (i + o), i <= (i + o), o ~ ((i + o) - i)) => RecurrentLayer (BasicRecurrent i o) ('D1 i) ('D1 o) where

  type RecTape (BasicRecurrent i o) ('D1 i) ('D1 o) = (S ('D1 o), S ('D1 i))
  -- Do a matrix vector multiplication and return the result.
  runRecurrentForwards (BasicRecurrent wB _ wN _) (S1D lastOutput) (S1D thisInput) =
    let thisOutput = S1D $ wB + wN #> (thisInput # lastOutput)
    in ((S1D lastOutput, S1D thisInput), thisOutput, thisOutput)

  -- Run a backpropogation step for a full connected layer.
  runRecurrentBackwards (BasicRecurrent _ _ wN _) (S1D lastOutput, S1D thisInput) (S1D dRec) (S1D dEdy) =
    let biasGradient        = (dRec + dEdy)
        layerGrad           = (dRec + dEdy) `outer` (thisInput # lastOutput)
        -- calcluate derivatives for next step
        (backGrad, recGrad) = split $ tr wN #> (dRec + dEdy)
    in  (BasicRecurrent' biasGradient layerGrad, S1D recGrad, S1D backGrad)


-------------------- GNum instances --------------------

instance (KnownNat i,KnownNat o,KnownNat (i+o)) => GNum (BasicRecurrent i o) where
  n |* (BasicRecurrent wB mB a nM) = (BasicRecurrent (fromRational n * wB) mB a nM)
  (BasicRecurrent wB mB a nM) |+ (BasicRecurrent wB2 mB2 a2 nM2)  = (BasicRecurrent (0.5 * (wB + wB2)) (0.5* (mB+mB2)) (0.5 * (a+a2)) (0.5 * (nM + nM2)))
  gFromRational r = (BasicRecurrent (fromRational r) 0 (fromRational r) 0)

