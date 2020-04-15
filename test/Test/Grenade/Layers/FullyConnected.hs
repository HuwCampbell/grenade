{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE CPP                 #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE KindSignatures      #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell     #-}
{-# LANGUAGE TypeOperators       #-}
{-# OPTIONS_GHC -fno-warn-missing-signatures #-}
module Test.Grenade.Layers.FullyConnected where

import           Data.Constraint               (Dict (..))
import           Data.Proxy
import           Data.Singletons               ()
import           GHC.TypeLits
import           Unsafe.Coerce                 (unsafeCoerce)

#if MIN_VERSION_base(4,9,0)
import           Data.Kind                     (Type)
#endif

import           Grenade.Core
import           Grenade.Layers.FullyConnected
import           Grenade.Utils.ListStore

import           Hedgehog

import           Test.Hedgehog.Compat
import           Test.Hedgehog.Hmatrix

data OpaqueFullyConnected :: Type where
     OpaqueFullyConnected :: (KnownNat i, KnownNat o, KnownNat (i * o)) => FullyConnected i o -> OpaqueFullyConnected

instance Show OpaqueFullyConnected where
    show (OpaqueFullyConnected n) = show n

genOpaqueFullyConnected :: Gen OpaqueFullyConnected
genOpaqueFullyConnected = do
    input   :: Integer  <- choose 2 100
    output  :: Integer  <- choose 1 100
    let Just input'      = someNatVal input
    let Just output'     = someNatVal output
    case (input', output') of
       (SomeNat (Proxy :: Proxy i'), SomeNat (Proxy :: Proxy o')) ->
         case (unsafeCoerce (Dict :: Dict ()) :: Dict (KnownNat (i' * o'))) of
           Dict -> do
            wB    <- randomVector
            bM    <- randomVector
            wN    <- uniformSample
            kM    <- uniformSample
            return . OpaqueFullyConnected $ (FullyConnected (FullyConnected' wB wN) (ListStore [Just $ FullyConnected' bM kM]) :: FullyConnected i' o')

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
