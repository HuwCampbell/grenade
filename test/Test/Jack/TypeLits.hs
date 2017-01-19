{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE ScopedTypeVariables   #-}

module Test.Jack.TypeLits where

import           Disorder.Jack

import           GHC.TypeLits

genNat :: Jack SomeNat
genNat = do
  Just n <- someNatVal <$> choose (1, 10)
  return n
