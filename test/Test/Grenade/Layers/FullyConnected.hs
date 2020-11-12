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
import           Data.Kind                     (Type)
import           Data.Proxy
import           Data.Singletons               ()
import qualified Data.Vector.Storable          as V
import           GHC.TypeLits
import           Hedgehog
import qualified Numeric.LinearAlgebra.Static  as LA
import           Test.Hedgehog.Compat
import           Test.Hedgehog.Hmatrix
import           Unsafe.Coerce                 (unsafeCoerce)

import           Grenade.Core
import           Grenade.Layers.FullyConnected
import           Grenade.Layers.Internal.BLAS
import           Grenade.Utils.ListStore

data OpaqueFullyConnected :: Type where
     OpaqueFullyConnected :: (KnownNat i, KnownNat o, KnownNat (i * o)) => FullyConnected i o -> OpaqueFullyConnected

instance Show OpaqueFullyConnected where
    show (OpaqueFullyConnected n) = show n

genOpaqueFullyConnected :: Gen OpaqueFullyConnected
genOpaqueFullyConnected = do
  input :: Integer <- choose 2 100
  output :: Integer <- choose 1 100
  variant <- choose 1 2
  let Just input' = someNatVal input
  let Just output' = someNatVal output
  case (input', output') of
    (SomeNat (Proxy :: Proxy i'), SomeNat (Proxy :: Proxy o')) ->
      case (unsafeCoerce (Dict :: Dict ()) :: Dict (KnownNat (i' * o'))) of
        Dict -> do
          wB <- randomVector
          bM <- randomVector
          wN <- uniformSample
          kM <- uniformSample
          let wInTmp = V.replicate (fromIntegral input) 0
              wBTmp = V.replicate (fromIntegral output) 0
              wNTmp = V.replicate (fromIntegral $ input * output) 0
          return . OpaqueFullyConnected $ case variant of
            1 -> (FullyConnected (FullyConnectedHMatrix wB wN) (ListStore 0 [Just $ FullyConnectedHMatrix bM kM]) (TempVectors wInTmp wBTmp wNTmp) :: FullyConnected i' o')
            2 ->
              let wB' = LA.extract wB
                  bM' = LA.extract bM
                  ext m = V.concat $ map LA.extract (LA.toRows m)
                  wN' = ext wN
                  kM' = ext kM
              in (FullyConnected (FullyConnectedBLAS wB' wN') (ListStore 0 [Just $ FullyConnectedBLAS bM' kM']) (TempVectors wInTmp wBTmp wNTmp) :: FullyConnected i' o')
            _ -> error "unexpected variant in genOpaqueFullyConnected in Test/FullyConnected.hs"

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
