{-# LANGUAGE CPP                 #-}
{-# LANGUAGE TemplateHaskell     #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE KindSignatures      #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# OPTIONS_GHC -fno-warn-missing-signatures #-}
module Test.Grenade.Layers.Pooling where

import           Data.Proxy
import           Data.Singletons ()

#if MIN_VERSION_base(4,9,0)
import           Data.Kind (Type)
#endif

import           GHC.TypeLits
import           Grenade.Layers.Pooling

import           Hedgehog

import           Test.Hedgehog.Compat

data OpaquePooling :: Type where
     OpaquePooling :: (KnownNat kh, KnownNat kw, KnownNat sh, KnownNat sw) => Pooling kh kw sh sw -> OpaquePooling

instance Show OpaquePooling where
    show (OpaquePooling n) = show n

genOpaquePooling :: Gen OpaquePooling
genOpaquePooling = do
    ~(Just kernelHeight) <- someNatVal <$> choose 2 15
    ~(Just kernelWidth ) <- someNatVal <$> choose 2 15
    ~(Just strideHeight) <- someNatVal <$> choose 2 15
    ~(Just strideWidth ) <- someNatVal <$> choose 2 15

    case (kernelHeight, kernelWidth, strideHeight, strideWidth) of
       (SomeNat (_ :: Proxy kh), SomeNat (_ :: Proxy kw), SomeNat (_ :: Proxy sh), SomeNat (_ :: Proxy sw)) ->
            return $ OpaquePooling (Pooling :: Pooling kh kw sh sw)

prop_pool_layer_witness =
  property $ do
    onet <- forAll genOpaquePooling
    case onet of
      (OpaquePooling (Pooling :: Pooling kernelRows kernelCols strideRows strideCols)) ->
        assert True

tests :: IO Bool
tests = checkParallel $$(discover)
