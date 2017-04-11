{-# LANGUAGE CPP                   #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE PolyKinds             #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE TemplateHaskell       #-}
{-# LANGUAGE FlexibleContexts      #-}
module Test.Hedgehog.Network where

import           Control.Monad.ST ( runST )

import           Data.Constraint
#if __GLASGOW_HASKELL__ < 800
import           Data.Proxy
#endif
import qualified Data.Vector.Storable as VS
import qualified Data.Vector.Storable.Mutable as VS ( write )
import           Data.Singletons
import           Data.Singletons.Prelude.List
-- import           Data.Type.Equality

import           Hedgehog
import           Hedgehog.Gen ( Gen )
import qualified Hedgehog.Gen as Gen
import           Hedgehog.Internal.Source
import           Hedgehog.Internal.Property ( failWith )

import           Grenade

import           GHC.TypeLits
import           GHC.TypeLits.Witnesses
import           Test.Hedgehog.Compat
import           Test.Hedgehog.TypeLits
import           Test.Hedgehog.Hmatrix
import           Test.Grenade.Layers.Convolution

import           Numeric.LinearAlgebra ( flatten )
import           Numeric.LinearAlgebra.Static ( size , extract, norm_Inf )
import           Unsafe.Coerce

data SomeNetwork :: * where
    SomeNetwork :: ( SingI shapes, SingI (Head shapes), SingI (Last shapes), Show (Network layers shapes) ) => Network layers shapes -> SomeNetwork

instance Show SomeNetwork where
  show (SomeNetwork net) = show net

-- | Generate a random network of a random type
--
-- This is slightly insane
genNetwork :: Monad m => Gen m SomeNetwork
genNetwork =
  Gen.recursive Gen.choice [
      do SomeSing ( r :: Sing final ) <- genShape
         withSingI r $
           pure (SomeNetwork (NNil :: Network '[] '[ final ] ))
    ] [
      do SomeNetwork ( rest :: Network layers shapes ) <- genNetwork
         case ( sing :: Sing shapes ) of
           SNil -> Gen.discard
           SCons ( r :: Sing h ) ( _ :: Sing hs ) ->
             withSingI r $ Gen.choice [
               pure (SomeNetwork (Logit   :~> rest :: Network ( Logit   ': layers ) ( h ': h ': hs )))
             , pure (SomeNetwork (Tanh    :~> rest :: Network ( Tanh    ': layers ) ( h ': h ': hs )))
             , case r of
                 D1Sing ->
                   Gen.choice [
                     pure (SomeNetwork (Relu    :~> rest :: Network ( Relu    ': layers ) ( h ': h ': hs )))
                   , pure (SomeNetwork (Elu     :~> rest :: Network ( Elu     ': layers ) ( h ': h ': hs )))
                   , pure (SomeNetwork (Softmax :~> rest :: Network ( Softmax ': layers ) ( h ': h ': hs )))
                   ]
                 D2Sing ->
                   Gen.choice [
                     pure (SomeNetwork (Relu    :~> rest :: Network ( Relu    ': layers ) ( h ': h ': hs )))
                   , pure (SomeNetwork (Elu     :~> rest :: Network ( Elu     ': layers ) ( h ': h ': hs )))
                   , do -- Build a convolution layer with one filter output
                        -- Figure out some kernel sizes which work for this layer
                        -- There must be a better way than this...
                        let dumb :: S h  = 0
                        case dumb of
                          S2D x -> do
                            let (rs, cs) = size x
                            let output_r = fromIntegral rs
                            let output_c = fromIntegral cs

                            let ok extent kernel = [stride | stride <- [ 1 .. extent ], (extent - kernel) `mod` stride == 0]

                            -- Get some kernels which will fit
                            kernel_r <- choose 1 output_r
                            kernel_c <- choose 1 output_c

                            -- Build up some strides which also fit
                            stride_r <- Gen.element $ ok output_r kernel_r
                            stride_c <- Gen.element $ ok output_c kernel_c

                            -- Determine the input size
                            let input_r = (output_r - 1) * stride_r + kernel_r
                            let input_c = (output_c - 1) * stride_c + kernel_c

                            -- Remake types for input
                            case ( someNatVal input_r, someNatVal input_c, someNatVal output_r, someNatVal output_c, someNatVal kernel_r, someNatVal kernel_c, someNatVal stride_r, someNatVal stride_c ) of
                              ( Just (SomeNat (_    :: Proxy inRows)),     Just (SomeNat  (_    :: Proxy inCols)),
                                Just (SomeNat (_    :: Proxy outRows)),    Just (SomeNat  (_    :: Proxy outCols)),
                                Just (SomeNat (pkr  :: Proxy kernelRows)), Just (SomeNat  (pkc  :: Proxy kernelCols)),
                                Just (SomeNat (_    :: Proxy strideRows)), Just (SomeNat  (_    :: Proxy strideCols))) ->
                                  let p1 = natDict pkr
                                      p2 = natDict pkc
                                  in  case ( p1 %* p2
                                            -- Fake it till you make it.
                                           , (unsafeCoerce (Dict :: Dict ()) :: Dict ( ( 'D2 outRows outCols ) ~ h ))
                                           , (unsafeCoerce (Dict :: Dict ()) :: Dict (((outRows - 1) * strideRows) ~ (inRows - kernelRows)))
                                           , (unsafeCoerce (Dict :: Dict ()) :: Dict (((outCols - 1) * strideCols) ~ (inCols - kernelCols)))) of
                                        (Dict, Dict, Dict, Dict) -> do
                                            conv <- genConvolution
                                            pure (SomeNetwork (conv :~> rest :: Network ( Convolution 1 1 kernelRows kernelCols strideRows strideCols ': layers ) ( ('D2 inRows inCols) ': h ': hs )))
                              _ -> Gen.discard
                   ]
                 D3Sing ->
                   Gen.choice [
                     pure (SomeNetwork (Relu    :~> rest :: Network ( Relu    ': layers ) ( h ': h ': hs )))
                   , pure (SomeNetwork (Elu     :~> rest :: Network ( Elu     ': layers ) ( h ': h ': hs )))
                   ]
             ]
    ]

-- Test a partial derivative numerically for a random network and input
prop_auto_diff :: Property
prop_auto_diff = property $ do
  SomeNetwork (network :: Network layers shapes) <- forAll genNetwork
  (input  :: S (Head shapes))     <- forAllRender nice genOfShape
  (target :: S (Last shapes))     <- forAllRender nice oneUp
  (tested :: S (Head shapes))     <- forAllRender nice oneUp

  let (tapes, output) = runNetwork network input
  let (_, bgrad)      = runGradient network tapes target
  let inputDiff       = input + tested * 0.00001
  let expected        = maxVal ( bgrad * tested )
  let (_, outputDiff) = runNetwork network inputDiff
  let result          = maxVal ( outputDiff * target - output * target ) / 0.00001

  result ~~~ expected

    where

-- Make a shape where all are 0 except for 1 value, which is 1.
oneUp :: forall shape m. ( Monad m, SingI shape ) => Gen m (S shape)
oneUp =
  case ( sing :: Sing shape ) of
    D1Sing -> let x = 0 :: S ( shape )
            in  case x of
                    ( S1D x' ) -> do
                    let ex = extract x'
                    let len = VS.length ex
                    ix <- choose 0 (len - 1)
                    let nx = runST $ do ex' <- VS.thaw ex
                                        VS.write ex' ix 1.0
                                        VS.freeze ex'
                    maybe Gen.discard pure . fromStorable $ nx

    D2Sing -> let x = 0 :: S ( shape )
            in  case x of
                    ( S2D x' ) -> do
                    let ex = flatten ( extract x' )
                    let len = VS.length ex
                    ix <- choose 0 (len - 1)
                    let nx = runST $ do ex' <- VS.thaw ex
                                        VS.write ex' ix 1
                                        VS.freeze ex'
                    maybe Gen.discard pure . fromStorable $ nx

    D3Sing -> let x = 0 :: S ( shape )
            in  case x of
                    ( S3D x' ) -> do
                    let ex = flatten ( extract x' )
                    let len = VS.length ex
                    ix <- choose 0 (len - 1)
                    let nx = runST $ do ex' <- VS.thaw ex
                                        VS.write ex' ix 1
                                        VS.freeze ex'
                    maybe Gen.discard pure . fromStorable $ nx

maxVal :: S shape -> Double
maxVal ( S1D x ) = norm_Inf x
maxVal ( S2D x ) = norm_Inf x
maxVal ( S3D x ) = norm_Inf x

(~~~) :: (Monad m, HasCallStack) => Double -> Double -> Test m ()
(~~~) x y =
  if (x - y) < 1e-5 then
    success
  else
    withFrozenCallStack $
    failWith Nothing $ unlines [
        "━━━ Not Simliar ━━━"
        , show x
        , show y
        ]
infix 4 ~~~


tests :: IO Bool
tests = $$(checkConcurrent)
