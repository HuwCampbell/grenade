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

import Data.Singletons.Prelude (Head, Last)

import Numeric.LinearAlgebra (maxIndex)
import Numeric.LinearAlgebra.Static (extract)

accuracy ::
       forall (shapes :: [Shape]) (layers :: [*]) (i :: Shape) (o :: Shape).
       (i ~ Head shapes, o ~ Last shapes)
    => Network layers shapes
    -> DataSet i o
    -> ClassificationAccuracy
accuracy net dataSet =
    let (inpt, labelShapes) = unzip dataSet
        outptShapes = runNet net <$> inpt
        nOfCorrectPredictions =
            length . filter (uncurry samePrediction) $
            zip labelShapes outptShapes
     in case constructProperFraction $
             fromIntegral nOfCorrectPredictions / fromIntegral (length dataSet) of
            Left errmess -> error errmess
            Right x -> x

samePrediction :: (S n) -> (S n) -> Bool
samePrediction (S1D label) (S1D prediction) =
    maxIndex (extract label) == maxIndex (extract prediction)
samePrediction _ _ =
    error "samePrediction takes one-dimensional shapes as arguments."
