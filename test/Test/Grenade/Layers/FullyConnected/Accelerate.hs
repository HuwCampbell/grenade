{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE TemplateHaskell     #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE KindSignatures      #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# OPTIONS_GHC -fno-warn-missing-signatures #-}
module Test.Grenade.Layers.FullyConnected.Accelerate where

import qualified Prelude as P
import           Prelude (Show, Maybe(..), IO, Integer, Monad, show, return, seq, (<$>))
import           Data.Proxy
import           Data.Singletons ()
import           Data.Array.Accelerate
import           Data.Array.Accelerate.Interpreter

import           GHC.TypeLits

import qualified Grenade.Core as G
import           Grenade.Core.Accelerate as A
import           Grenade.Layers.FullyConnected

import           Hedgehog

import           Test.Hedgehog.Compat
import           Test.Hedgehog.Hmatrix
import           Test.Hedgehog.Accelerate

data OpaqueFullyConnected :: * where
     OpaqueFullyConnected :: (KnownNat i, KnownNat o) => FullyConnected i o -> OpaqueFullyConnected

instance Show OpaqueFullyConnected where
    show (OpaqueFullyConnected n) = show n

genOpaqueFullyConnected :: Monad m => Gen m OpaqueFullyConnected
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
    input :: S DIM1 <- blindForAll (randomArray (Z :. (P.fromIntegral $ natVal (Proxy :: Proxy i))))
    let (fclayer' :: AFullyConnected i o) = toAccel fclayer
        (tape, output :: Acc (S DIM1)) = runForwards fclayer' (use input)
        backed :: (Gradient (AFullyConnected i o), Acc (S DIM1))
                                    = runBackwards fclayer' tape output
    (run $ lift backed) `seq` success

(~===) :: (P.Num (Exp e), P.Fractional (Exp e), RealFrac e, Monad m, P.Eq sh, P.Eq e, Elt e, Shape sh, FromIntegral Int e) => Array sh e -> Array sh e -> Test m ()
a ~=== b = fuzzy a === fuzzy b
  where
    fuzzy :: (P.Num (Exp e), P.Fractional (Exp e), RealFrac e, Shape sh, Elt e, FromIntegral Int e) => Array sh e -> Array sh e
    fuzzy = run1 $ map $ \x ->
      let
        scaledUp :: Exp Int
        scaledUp = round $ x * 1e7
      in (fromIntegral scaledUp) / 1e7

prop_fully_connected_forwards_equals_reference :: Property
prop_fully_connected_forwards_equals_reference = property $ do
    OpaqueFullyConnected (fclayer :: FullyConnected i o) <- blindForAll genOpaqueFullyConnected
    input :: G.S ('G.D1 i) <- blindForAll (G.S1D <$> randomVector)


    let
      input' = case input of
        G.S1D v -> fromVector v
      output :: G.S ('G.D1 o)
      inputGrad :: G.S ('G.D1 i)
      (tape, output) = G.runForwards fclayer input
      (gradient, inputGrad) = G.runBackwards fclayer tape output

      tapeV :: Vector Double
      tapeV = case tape of
        G.S1D v -> fromVector v

      outputV :: Vector Double
      outputV = case output of
        G.S1D v -> fromVector v

      inputGradV :: Vector Double
      inputGradV = case inputGrad of
        G.S1D v -> fromVector v

      biasGradV :: Vector Double
      actGradV :: Array DIM2 Double
      (biasGradV, actGradV) = case gradient of
        FullyConnected' b a -> (fromVector b, fromMatrix a)

      (fclayer' :: AFullyConnected i o) = toAccel fclayer
      (tape', output') = runForwards fclayer' (use input')
      (gradient', inputGrad') = runBackwards fclayer' tape' output'

      (tapeV', outputV', (biasGradV', actGradV'), inputGradV') = run $ lift (tape', output', gradient', inputGrad')
    tapeV ~=== tapeV'
    outputV ~=== outputV'
    biasGradV ~=== biasGradV'
    actGradV ~=== actGradV'
    inputGradV ~=== inputGradV'

tests :: IO Bool
tests = checkParallel $$(discover)
