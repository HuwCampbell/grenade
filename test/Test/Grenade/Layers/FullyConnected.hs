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

import           Grenade.Core.Shape
import           Grenade.Core.Network
import           Grenade.Layers.FullyConnected

import qualified Numeric.LinearAlgebra.Static as HStatic

import           Disorder.Jack

import           Test.Jack.Hmatrix


data OpaqueFullyConnected :: * where
     OpaqueFullyConnected :: (KnownNat i, KnownNat o) => FullyConnected i o -> OpaqueFullyConnected

instance Show OpaqueFullyConnected where
    show (OpaqueFullyConnected n) = show n

genOpaqueFullyConnected :: Jack OpaqueFullyConnected
genOpaqueFullyConnected = do
    input   :: Integer  <- choose (2, 100)
    output  :: Integer  <- choose (2, 100)
    let Just input'      = someNatVal input
    let Just output'     = someNatVal output
    case (input', output') of
       (SomeNat (Proxy :: Proxy i'), SomeNat (Proxy :: Proxy o')) -> do
            wB    <- randomVector
            bM    <- randomVector
            wN    <- uniformSample
            kM    <- uniformSample
            return . OpaqueFullyConnected $ (FullyConnected wB bM wN kM :: FullyConnected i' o')

prop_fully_connected_forwards :: Property
prop_fully_connected_forwards =
    gamble genOpaqueFullyConnected $ \(OpaqueFullyConnected (fclayer :: FullyConnected i o)) ->
        let i = fromIntegral $ natVal (Proxy :: Proxy i)
        in gamble (vectorOf i sizedRealFrac) $ \input ->
            let x :: S' ('D1 o) = runForwards fclayer (S1D' (HStatic.vector input :: HStatic.R i))
            in  x `seq` True


return []
tests :: IO Bool
tests = $quickCheckAll
