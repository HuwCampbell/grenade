{-# LANGUAGE CPP                   #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE UndecidableInstances  #-}

module Grenade.Recurrent.Layers.BasicRecurrent (
    BasicRecurrent (..)
  , randomBasicRecurrent
  ) where



import           Control.Monad.Random ( MonadRandom, getRandom )

import           Data.Singletons.TypeLits

#if MIN_VERSION_base(4,9,0)
import           Data.Kind (Type)
#endif

import           Numeric.LinearAlgebra.Static

import           GHC.TypeLits

import           Grenade.Core
import           Grenade.Recurrent.Core

data BasicRecurrent :: Nat -- Input layer size
                    -> Nat -- Output layer size
                    -> Type where
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
                     -> Type where
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

  runUpdate lp (BasicRecurrent oldBias oldBiasMomentum oldActivations oldMomentum) (BasicRecurrent' biasGradient activationGradient) =
    let newBiasMomentum = konst (learningMomentum lp) * oldBiasMomentum - konst (learningRate lp) * biasGradient
        newBias         = oldBias + newBiasMomentum
        newMomentum     = konst (learningMomentum lp) * oldMomentum - konst (learningRate lp) * activationGradient
        regulariser     = konst (learningRegulariser lp * learningRate lp) * oldActivations
        newActivations  = oldActivations + newMomentum - regulariser
    in BasicRecurrent newBias newBiasMomentum newActivations newMomentum

  createRandom = randomBasicRecurrent

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

randomBasicRecurrent :: (MonadRandom m, KnownNat i, KnownNat o, KnownNat x, x ~ (i + o))
                     => m (BasicRecurrent i o)
randomBasicRecurrent = do
    seed1 <- getRandom
    seed2 <- getRandom
    let wB = randomVector seed1 Uniform * 2 - 1
        wN = uniformSample seed2 (-1) 1
        bm = konst 0
        mm = konst 0
    return $ BasicRecurrent wB bm wN mm
