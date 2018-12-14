{-# LANGUAGE CPP                 #-}
{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE TemplateHaskell     #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE KindSignatures      #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# OPTIONS_GHC -fno-warn-missing-signatures #-}
module Test.Grenade.Layers.FullyConnected where

import           Data.Proxy
import           Data.Singletons ()

import           GHC.TypeLits

#if MIN_VERSION_base(4,9,0)
import           Data.Kind (Type)
#endif

import           Grenade.Core
import           Grenade.Layers.FullyConnected

import           Hedgehog

import           Test.Hedgehog.Compat
import           Test.Hedgehog.Hmatrix

data OpaqueFullyConnected :: Type where
     OpaqueFullyConnected :: (KnownNat i, KnownNat o) => FullyConnected i o -> OpaqueFullyConnected

instance Show OpaqueFullyConnected where
    show (OpaqueFullyConnected n) = show n

genOpaqueFullyConnected :: Gen OpaqueFullyConnected
genOpaqueFullyConnected = do
    input   :: Integer  <- choose 2 100
    output  :: Integer  <- choose 1 100
    let Just input'      = someNatVal input
    let Just output'     = someNatVal output
    case (input', output') of
       (SomeNat (Proxy :: Proxy i'), SomeNat (Proxy :: Proxy o')) -> do
            wB    <- randomVector
            bM    <- randomVector
            wN    <- uniformSample
            kM    <- uniformSample
            return . OpaqueFullyConnected $ (FullyConnected (FullyConnected' wB wN) (FullyConnected' bM kM) :: FullyConnected i' o')

prop_fully_connected_forwards :: Property
prop_fully_connected_forwards = property $ do
    OpaqueFullyConnected (fclayer :: FullyConnected i o) <- blindForAll genOpaqueFullyConnected
    input :: S ('D1 i) <- blindForAll (S1D <$> randomVector)
    let (tape, output :: S ('D1 o)) = runForwards fclayer input
        backed :: (Gradient (FullyConnected i o), S ('D1 i))
                                    = runBackwards fclayer tape output
    backed `seq` success

tests :: IO Bool
tests = checkParallel $$(discover)
