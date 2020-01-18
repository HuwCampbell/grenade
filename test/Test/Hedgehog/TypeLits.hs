{-# LANGUAGE CPP                   #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE PolyKinds             #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE GADTs                 #-}
module Test.Hedgehog.TypeLits where

import           Data.Constraint (Dict (..))
import           Data.Singletons (Proxy (..), Sing (..), SomeSing (..), sing)
#if MIN_VERSION_singletons(2,6,0)
import           Data.Singletons.TypeLits (SNat (..))
#endif

import           Hedgehog (Gen)
import qualified Hedgehog.Gen as Gen

import           Grenade

import           GHC.TypeLits (SomeNat (..), natVal, someNatVal)
import           GHC.TypeLits.Witnesses ((%*), natDict)
import           Test.Hedgehog.Compat (choose)

genNat :: Gen SomeNat
genNat = do
  ~(Just n) <- someNatVal <$> choose 1 10
  return n

genShape :: Gen (SomeSing Shape)
genShape
  = Gen.choice [
      genD1
    , genD2
    , genD3
    ]

genD1 :: Gen (SomeSing Shape)
genD1 = do
  n <- genNat
  return $ case n of
    SomeNat (_ :: Proxy x) -> SomeSing (sing :: Sing ('D1 x))

genD2 :: Gen (SomeSing Shape)
genD2 = do
  n <- genNat
  m <- genNat
  return $ case (n, m) of
    (SomeNat (_ :: Proxy x), SomeNat (_ :: Proxy y)) -> SomeSing (sing :: Sing ('D2 x y))

genD3 :: Gen (SomeSing Shape)
genD3 = do
  n <- genNat
  m <- genNat
  o <- genNat
  return $ case (n, m, o) of
    (SomeNat (px :: Proxy x), SomeNat (_ :: Proxy y), SomeNat (pz :: Proxy z)) ->
        case natDict px %* natDict pz of
          Dict -> SomeSing (sing :: Sing ('D3 x y z))

rss :: SomeSing Shape -> String
rss (SomeSing (r :: Sing s)) = case r of
  (D1Sing a@SNat) -> "D1 " ++ show (natVal a)
  (D2Sing a@SNat b@SNat) -> "D2 " ++ show (natVal a) ++ " " ++ show (natVal b)
  (D3Sing a@SNat b@SNat c@SNat) -> "D3 " ++ show (natVal a) ++ " " ++ show (natVal b) ++ " " ++ show (natVal c)
