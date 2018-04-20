{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}

module Grenade.Train.Test
    ( accuracy
    ) where

import Grenade.Core
import Grenade.Train.DataSet
import Grenade.Utils.Accuracy

import Control.Monad.Catch (displayException)

import Data.Singletons.Prelude (Head, Last)

import Numeric.LinearAlgebra (maxIndex)
import Numeric.LinearAlgebra.Static (extract)

accuracy
    :: forall (shapes :: [Shape]) (layers :: [*]) (i :: Shape) (o :: Shape)
    . (i ~ Head shapes, o ~ Last shapes)
    => Network layers shapes -> DataSet i o -> Accuracy
accuracy net dataSet =
    let (inpt, labelShapes) = unzip dataSet
        outptShapes = runNet net <$> inpt
        nOfCorrectPredictions = length . filter (uncurry samePrediction) $ zip labelShapes outptShapes
    in case accuracyM $ fromIntegral nOfCorrectPredictions / fromIntegral (length dataSet) of
        Right a -> a
        Left e -> error $ displayException e

samePrediction :: (S n) -> (S n) -> Bool
samePrediction (S1D label) (S1D prediction) = maxIndex (extract label) == maxIndex (extract prediction)
samePrediction _ _ = error "NeuralNet.Test.samePrediction: The output and label are not 1D vectors"
