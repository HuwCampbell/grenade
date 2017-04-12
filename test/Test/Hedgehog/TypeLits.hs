{-# LANGUAGE CPP                   #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE PolyKinds             #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE GADTs                 #-}
module Test.Hedgehog.TypeLits where

import           Data.Constraint
#if __GLASGOW_HASKELL__ < 800
import           Data.Proxy
#endif
import           Data.Singletons

import qualified Hedgehog.Gen as Gen

import           Grenade

import           GHC.TypeLits
import           GHC.TypeLits.Witnesses
import           Test.Hedgehog.Compat

genNat :: Monad m => Gen.Gen m SomeNat
genNat = do
  Just n <- someNatVal <$> choose 1 10
  return n

#if __GLASGOW_HASKELL__ < 800
type Shape' = ('KProxy :: KProxy Shape)
#else
type Shape' = Shape
#endif

genShape :: Monad m => Gen.Gen m (SomeSing Shape')
genShape
  = Gen.choice [
      genD1
    , genD2
    , genD3
    ]

genD1 :: Monad m => Gen.Gen m (SomeSing Shape')
genD1 = do
  n <- genNat
  return $ case n of
    SomeNat (_ :: Proxy x) -> SomeSing (sing :: Sing ('D1 x))

genD2 :: Monad m => Gen.Gen m (SomeSing Shape')
genD2 = do
  n <- genNat
  m <- genNat
  return $ case (n, m) of
    (SomeNat (_ :: Proxy x), SomeNat (_ :: Proxy y)) -> SomeSing (sing :: Sing ('D2 x y))

genD3 :: Monad m => Gen.Gen m (SomeSing Shape')
genD3 = do
  n <- genNat
  m <- genNat
  o <- genNat
  return $ case (n, m, o) of
    (SomeNat (px :: Proxy x), SomeNat (_ :: Proxy y), SomeNat (pz :: Proxy z)) ->
        case natDict px %* natDict pz of
          Dict -> SomeSing (sing :: Sing ('D3 x y z))
