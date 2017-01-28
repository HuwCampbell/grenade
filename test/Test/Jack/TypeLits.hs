{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE ScopedTypeVariables   #-}

module Test.Jack.TypeLits where

import           Data.Constraint
import           Data.Singletons
import           Disorder.Jack

import           Grenade

import           GHC.TypeLits
import           GHC.TypeLits.Witnesses

genNat :: Jack SomeNat
genNat = do
  Just n <- someNatVal <$> choose (1, 10)
  return n

genShape :: Jack (SomeSing Shape)
genShape
  = oneOf [
      genD1
    , genD2
    , genD3
    ]
  where
    genD1 = do
      n <- genNat
      return $ case n of
        SomeNat (_ :: Proxy x) -> SomeSing (sing :: Sing ('D1 x))

    genD2 = do
      n <- genNat
      m <- genNat
      return $ case (n, m) of
        (SomeNat (_ :: Proxy x), SomeNat (_ :: Proxy y)) -> SomeSing (sing :: Sing ('D2 x y))

    genD3 = do
      n <- genNat
      m <- genNat
      o <- genNat
      return $ case (n, m, o) of
        (SomeNat (px :: Proxy x), SomeNat (_ :: Proxy y), SomeNat (pz :: Proxy z)) ->
           case natDict px %* natDict pz of
              Dict -> SomeSing (sing :: Sing ('D3 x y z))
