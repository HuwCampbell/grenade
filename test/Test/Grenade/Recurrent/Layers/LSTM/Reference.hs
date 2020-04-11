{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE ConstraintKinds       #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE DeriveFunctor         #-}
{-# LANGUAGE DeriveFoldable        #-}
{-# LANGUAGE DeriveTraversable     #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE RankNTypes            #-}

{-# OPTIONS_GHC -fno-warn-missing-signatures #-}
module Test.Grenade.Recurrent.Layers.LSTM.Reference where

import           Data.Reflection
import           Numeric.AD.Mode.Reverse
import           Numeric.AD.Internal.Reverse (Tape)

import           GHC.TypeLits (KnownNat)

import           Grenade.Recurrent.Layers.LSTM (LSTMWeights (..))
import qualified Grenade.Recurrent.Layers.LSTM as LSTM

import qualified Numeric.LinearAlgebra.Static as S
import qualified Numeric.LinearAlgebra as H

--
-- This module contains a set of list only versions of
-- an LSTM layer which can be used with the AD library.
--
-- Using this, we can check to make sure that our fast
-- back propagation implementation is correct.
--

-- | List only matrix deriving functor
data Matrix a = Matrix {
    matrixWeights         :: [[a]]
  } deriving (Functor, Foldable, Traversable, Eq, Show)

-- | List only vector deriving functor
data Vector a = Vector {
    vectorWeights         :: [a]
  } deriving (Functor, Foldable, Traversable, Eq, Show)

-- | List only LSTM weights
data RefLSTM a = RefLSTM
    { refLstmWf :: Matrix a -- Weight Forget     (W_f)
    , refLstmUf :: Matrix a -- Cell State Forget (U_f)
    , refLstmBf :: Vector a -- Bias Forget       (b_f)
    , refLstmWi :: Matrix a -- Weight Input      (W_i)
    , refLstmUi :: Matrix a -- Cell State Input  (U_i)
    , refLstmBi :: Vector a -- Bias Input        (b_i)
    , refLstmWo :: Matrix a -- Weight Output     (W_o)
    , refLstmUo :: Matrix a -- Cell State Output (U_o)
    , refLstmBo :: Vector a -- Bias Output       (b_o)
    , refLstmWc :: Matrix a -- Weight Cell       (W_c)
    , refLstmBc :: Vector a -- Bias Cell         (b_c)
    } deriving (Functor, Foldable, Traversable, Eq, Show)

lstmToReference :: (KnownNat a, KnownNat b) => LSTM.LSTMWeights a b -> RefLSTM Double
lstmToReference lw =
    RefLSTM
      { refLstmWf = Matrix . H.toLists . S.extract $ lstmWf lw -- Weight Forget     (W_f)
      , refLstmUf = Matrix . H.toLists . S.extract $ lstmUf lw -- Cell State Forget (U_f)
      , refLstmBf = Vector . H.toList  . S.extract $ lstmBf lw -- Bias Forget       (b_f)
      , refLstmWi = Matrix . H.toLists . S.extract $ lstmWi lw -- Weight Input      (W_i)
      , refLstmUi = Matrix . H.toLists . S.extract $ lstmUi lw -- Cell State Input  (U_i)
      , refLstmBi = Vector . H.toList  . S.extract $ lstmBi lw -- Bias Input        (b_i)
      , refLstmWo = Matrix . H.toLists . S.extract $ lstmWo lw -- Weight Output     (W_o)
      , refLstmUo = Matrix . H.toLists . S.extract $ lstmUo lw -- Cell State Output (U_o)
      , refLstmBo = Vector . H.toList  . S.extract $ lstmBo lw -- Bias Output       (b_o)
      , refLstmWc = Matrix . H.toLists . S.extract $ lstmWc lw -- Weight Cell       (W_c)
      , refLstmBc = Vector . H.toList  . S.extract $ lstmBc lw -- Bias Cell         (b_c)
      }

runLSTM :: Floating a => RefLSTM a -> Vector a -> Vector a -> (Vector a, Vector a)
runLSTM rl cell input =
    let -- Forget state vector
        f_t = sigmoid   $ refLstmBf rl #+ refLstmWf rl #> input #+ refLstmUf rl #> cell
        -- Input state vector
        i_t = sigmoid   $ refLstmBi rl #+ refLstmWi rl #> input #+ refLstmUi rl #> cell
        -- Output state vector
        o_t = sigmoid   $ refLstmBo rl #+ refLstmWo rl #> input #+ refLstmUo rl #> cell
        -- Cell input state vector
        c_x = fmap tanh $ refLstmBc rl #+ refLstmWc rl #> input
        -- Cell state
        c_t = f_t #* cell #+ i_t #* c_x
        -- Output (it's sometimes recommended to use tanh c_t)
        h_t = o_t #* c_t
    in (c_t, h_t)

runLSTMback :: forall a. Floating a => Vector a -> Vector a -> RefLSTM a -> RefLSTM a
runLSTMback cell input =
  grad f
    where
  f :: forall s. Reifies s Tape => RefLSTM (Reverse s a) -> Reverse s a
  f net =
    let cell'   = fmap auto cell
        input'  = fmap auto input
        (cells, forwarded) = runLSTM net cell' input'
    in  sum forwarded + sum cells

runLSTMbackOnInput :: forall a. Floating a => Vector a -> RefLSTM a -> Vector a -> Vector a
runLSTMbackOnInput cell net =
  grad f
    where
  f :: forall s. Reifies s Tape => Vector (Reverse s a) -> Reverse s a
  f input =
    let cell'   = fmap auto cell
        net'    = fmap auto net
        (cells, forwarded) = runLSTM net' cell' input
    in  sum forwarded + sum cells

runLSTMbackOnCell :: forall a. Floating a => Vector a -> RefLSTM a -> Vector a -> Vector a
runLSTMbackOnCell input net =
  grad f
    where
  f :: forall s. Reifies s Tape => Vector (Reverse s a) -> Reverse s a
  f cell =
    let input'  = fmap auto input
        net'    = fmap auto net
        (cells, forwarded) = runLSTM net' cell input'
    in  sum forwarded + sum cells

-- | Helper to multiply a matrix by a vector
matMult :: Num a => Matrix a -> Vector a -> Vector a
matMult (Matrix m) (Vector v) = Vector result
  where
    lrs = map length m
    l   = length v
    result = if all (== l) lrs
             then map (\r -> sum $ zipWith (*) r v) m
             else error $ "Matrix has rows of length " ++ show lrs ++
                          " but vector is of length " ++ show l

(#>) :: Num a => Matrix a -> Vector a -> Vector a
(#>) = matMult
infixr 8 #>

(#+) :: Num a => Vector a -> Vector a -> Vector a
(#+) (Vector as) (Vector bs) = Vector $ zipWith (+) as bs
infixl 6 #+

(#-) :: Num a => Vector a -> Vector a -> Vector a
(#-) (Vector as) (Vector bs) = Vector $ zipWith (-) as bs
infixl 6 #-

(#*) :: Num a => Vector a -> Vector a -> Vector a
(#*) (Vector as) (Vector bs) = Vector $ zipWith (*) as bs
infixl 7 #*

sigmoid :: (Functor f, Floating a) => f a -> f a
sigmoid xs = (\x -> 1 / (1 + exp (-x))) <$> xs
