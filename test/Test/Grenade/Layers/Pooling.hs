{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE GADTs             #-}
{-# OPTIONS_GHC -fno-warn-missing-signatures #-}
module Test.Grenade.Layers.Pooling where

import           Grenade.Layers.Internal.Pooling

import           Numeric.LinearAlgebra hiding (uniformSample, konst, (===))

import           Test.QuickCheck hiding ((><))

prop_pool = once $
 let input = (3><4)
               [ 1.0, 14.0,  3.0,  4.0
               , 5.0,  6.0,  7.0,  8.0
               , 9.0, 10.0, 11.0, 12.0 ]
     expected = (2><3)
               [ 14.0,  14.0,  8.0
               , 10.0, 11.0, 12.0 ]
     out = poolForward 2 2 1 1 2 3 input
 in expected === out

prop_pool_backwards = once $
 let input = (3><4)
               [ 1.0,  2.0,  3.0,  4.0
               , 5.0,  6.0,  7.0,  8.0
               , 9.0, 10.0, 11.0, 12.0 ]
     grads = (2><3)
               [ -6.0,   -7.0,  -8.0
               , -10.0, -11.0, -12.0 ]
     expected = (3><4)
               [ 0.0,  0.0,  0.0,  0.0
               , 0.0, -6.0, -7.0, -8.0
               , 0.0,-10.0,-11.0,-12.0 ]
     out = poolBackward 2 2 1 1 input grads
 in expected === out

prop_pool_backwards_additive = once $
 let input = (3><4)
               [ 4.0,  2.0,  3.0,  4.0
               , 0.0,  0.0,  7.0,  8.0
               , 9.0,  0.0,  0.0,  0.0 ]
     grads = (2><3)
               [ -6.0,   -7.0,  -8.0
               , -10.0, -11.0, -12.0 ]
     expected = (3><4)
               [-6.0,  0.0,  0.0,  0.0
               , 0.0,  0.0,-18.0,-20.0
               ,-10.0, 0.0,  0.0,  0.0 ]
     out = poolBackward 2 2 1 1 input grads
 in expected === out

return []
tests :: IO Bool
tests = $quickCheckAll
