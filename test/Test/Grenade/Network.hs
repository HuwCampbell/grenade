{-# LANGUAGE CPP                   #-}
{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE PolyKinds             #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE TemplateHaskell       #-}
{-# LANGUAGE FlexibleContexts      #-}
module Test.Grenade.Network where

import           Control.Monad ( guard )
import           Control.Monad.ST ( runST )

import           Data.Constraint
#if __GLASGOW_HASKELL__ < 800
import           Data.Proxy
#endif
import qualified Data.Vector.Storable as VS
import qualified Data.Vector.Storable.Mutable as VS ( write )
import           Data.Singletons
import           Data.Singletons.Prelude.List
import           Data.Singletons.TypeLits

-- import           Data.Type.Equality

import           Hedgehog
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
import           Numeric.LinearAlgebra.Static ( extract, norm_Inf )
import           Unsafe.Coerce

data SomeNetwork :: * where
    SomeNetwork :: ( SingI shapes, SingI (Head shapes), SingI (Last shapes), Show (Network layers shapes) ) => Network layers shapes -> SomeNetwork

instance Show SomeNetwork where
  show (SomeNetwork net) = show net

-- | Generate a random network of a random type
--
-- This is slightly insane for a few reasons. Everything must be wrapped up
-- in a SomeNetwork.
genNetwork :: Monad m => Gen.Gen m SomeNetwork
genNetwork =
  Gen.recursive Gen.choice [
      do SomeSing ( r :: Sing final ) <- genShape
         withSingI r $
           pure (SomeNetwork (NNil :: Network '[] '[ final ] ))
    ] [
      do SomeNetwork ( rest :: Network layers shapes ) <- genNetwork
         case ( sing :: Sing shapes ) of
           SNil -> Gen.discard -- Can't occur
           SCons ( h :: Sing h ) ( _ :: Sing hs ) ->
             withSingI h $
              case h of
                 D1Sing l@SNat ->
                   Gen.choice [
                     pure (SomeNetwork (Tanh    :~> rest :: Network ( Tanh    ': layers ) ( h ': h ': hs )))
                   , pure (SomeNetwork (Logit   :~> rest :: Network ( Logit   ': layers ) ( h ': h ': hs )))
                   , pure (SomeNetwork (Relu    :~> rest :: Network ( Relu    ': layers ) ( h ': h ': hs )))
                   , pure (SomeNetwork (Elu     :~> rest :: Network ( Elu     ': layers ) ( h ': h ': hs )))
                   , pure (SomeNetwork (Softmax :~> rest :: Network ( Softmax ': layers ) ( h ': h ': hs )))
                   , do -- Reshape to two dimensions
                        let divisors n = 1 : [x | x <- [2..(n-1)], n `rem` x == 0]
                        let len = natVal l
                        rs  <- Gen.element $ divisors len
                        let cs  = len `quot` rs
                        case ( someNatVal rs, someNatVal cs, someNatVal len ) of
                          ( Just (SomeNat (rs' :: Proxy inRows)), Just (SomeNat (cs' :: Proxy inCols)), Just (SomeNat (_ :: Proxy outLen ) )) ->
                            let p1 = natDict rs'
                                p2 = natDict cs'
                            in  case ( p1 %* p2, unsafeCoerce (Dict :: Dict ()) :: Dict ((inRows * inCols) ~ outLen), unsafeCoerce (Dict :: Dict ()) :: Dict (( 'D1 outLen ) ~ h )) of
                                  ( Dict, Dict, Dict ) ->
                                    pure (SomeNetwork (Reshape :~> rest :: Network ( Reshape ': layers ) ( ('D2 inRows inCols) ': h ': hs )))
                          _ -> Gen.discard -- Doesn't occur
                   ]
                 D2Sing r@SNat c@SNat ->
                   Gen.choice [
                     pure (SomeNetwork (Tanh    :~> rest :: Network ( Tanh    ': layers ) ( h ': h ': hs )))
                   , pure (SomeNetwork (Logit   :~> rest :: Network ( Logit   ': layers ) ( h ': h ': hs )))
                   , pure (SomeNetwork (Relu    :~> rest :: Network ( Relu    ': layers ) ( h ': h ': hs )))
                   , pure (SomeNetwork (Elu     :~> rest :: Network ( Elu     ': layers ) ( h ': h ': hs )))
                   , do -- Build a convolution layer with one filter output
                        -- Figure out some kernel sizes which work for this layer
                        -- There must be a better way than this...
                        let output_r = natVal r
                        let output_c = natVal c

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

                        guard (input_r < 100)
                        guard (input_c < 100)

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
                          _ -> Gen.discard -- Can't occur
                   , do -- Build a convolution layer with one filter output
                        -- Figure out some kernel sizes which work for this layer
                        -- There must be a better way than this...
                        let output_r = natVal r
                        let output_c = natVal c

                        let ok extent kernel = [stride | stride <- [ 1 .. extent ], (extent - kernel) `mod` stride == 0]

                        -- Get some kernels which will fit
                        kernel_r <- choose 1 output_r
                        kernel_c <- choose 1 output_c
                        channels <- choose 1 10

                        -- Build up some strides which also fit
                        stride_r <- Gen.element $ ok output_r kernel_r
                        stride_c <- Gen.element $ ok output_c kernel_c

                        -- Determine the input size
                        let input_r = (output_r - 1) * stride_r + kernel_r
                        let input_c = (output_c - 1) * stride_c + kernel_c

                        guard (input_r < 100)
                        guard (input_c < 100)

                        -- Remake types for input
                        case ( someNatVal channels, someNatVal input_r, someNatVal input_c, someNatVal output_r, someNatVal output_c, someNatVal kernel_r, someNatVal kernel_c, someNatVal stride_r, someNatVal stride_c ) of
                          ( Just (SomeNat (chan :: Proxy channels)),
                            Just (SomeNat (pinr :: Proxy inRows)),     Just (SomeNat  (_    :: Proxy inCols)),
                            Just (SomeNat (_    :: Proxy outRows)),    Just (SomeNat  (_    :: Proxy outCols)),
                            Just (SomeNat (pkr  :: Proxy kernelRows)), Just (SomeNat  (pkc  :: Proxy kernelCols)),
                            Just (SomeNat (_    :: Proxy strideRows)), Just (SomeNat  (_    :: Proxy strideCols))) ->
                              let p1 = natDict pkr
                                  p2 = natDict pkc
                                  p3 = natDict chan
                              in  case ( p1 %* p2 %* p3
                                        , natDict pinr %* natDict chan
                                        -- Fake it till you make it.
                                        , (unsafeCoerce (Dict :: Dict ()) :: Dict ( ( 'D2 outRows outCols ) ~ h ))
                                        , (unsafeCoerce (Dict :: Dict ()) :: Dict (((outRows - 1) * strideRows) ~ (inRows - kernelRows)))
                                        , (unsafeCoerce (Dict :: Dict ()) :: Dict (((outCols - 1) * strideCols) ~ (inCols - kernelCols)))) of
                                    (Dict, Dict, Dict, Dict, Dict) -> do
                                        conv <- genConvolution
                                        pure (SomeNetwork (conv :~> rest :: Network ( Convolution channels 1 kernelRows kernelCols strideRows strideCols ': layers ) ( ('D3 inRows inCols channels) ': h ': hs )))
                          _ -> Gen.discard -- Can't occur
                   , do -- Build a Pooling layer
                        let output_r = natVal r
                        let output_c = natVal c

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

                        guard (input_r < 100)
                        guard (input_c < 100)

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
                                    (Dict, Dict, Dict, Dict) ->
                                        pure (SomeNetwork (Pooling :~> rest :: Network ( Pooling kernelRows kernelCols strideRows strideCols ': layers ) ( ('D2 inRows inCols) ': h ': hs )))
                          _ -> Gen.discard -- Can't occur
                   ]
                 D3Sing r@SNat c@SNat f@SNat ->
                   Gen.choice [
                     pure (SomeNetwork (Tanh    :~> rest :: Network ( Tanh    ': layers ) ( h ': h ': hs )))
                   , pure (SomeNetwork (Logit   :~> rest :: Network ( Logit   ': layers ) ( h ': h ': hs )))
                   , pure (SomeNetwork (Relu    :~> rest :: Network ( Relu    ': layers ) ( h ': h ': hs )))
                   , pure (SomeNetwork (Elu     :~> rest :: Network ( Elu     ': layers ) ( h ': h ': hs )))
                   , do -- Build a convolution layer with one filter output
                        -- Figure out some kernel sizes which work for this layer
                        -- There must be a better way than this...
                        let output_r = natVal r
                        let output_c = natVal c
                        let output_f = natVal f

                        let ok extent kernel = [stride | stride <- [ 1 .. extent ], (extent - kernel) `mod` stride == 0]

                        -- Get some kernels which will fit
                        kernel_r <- choose 1 output_r
                        kernel_c <- choose 1 output_c
                        channels <- choose 1 10

                        -- Build up some strides which also fit
                        stride_r <- Gen.element $ ok output_r kernel_r
                        stride_c <- Gen.element $ ok output_c kernel_c

                        -- Determine the input size
                        let input_r = (output_r - 1) * stride_r + kernel_r
                        let input_c = (output_c - 1) * stride_c + kernel_c

                        guard (input_r < 100)
                        guard (input_c < 100)

                        -- Remake types for input
                        case ( someNatVal channels, someNatVal output_f, someNatVal input_r, someNatVal input_c, someNatVal output_r, someNatVal output_c, someNatVal kernel_r, someNatVal kernel_c, someNatVal stride_r, someNatVal stride_c ) of
                          ( Just (SomeNat (chan :: Proxy channels)),   Just (SomeNat  (_    :: Proxy filters)),
                            Just (SomeNat (pinr :: Proxy inRows)),     Just (SomeNat  (_    :: Proxy inCols)),
                            Just (SomeNat (_    :: Proxy outRows)),    Just (SomeNat  (_    :: Proxy outCols)),
                            Just (SomeNat (pkr  :: Proxy kernelRows)), Just (SomeNat  (pkc  :: Proxy kernelCols)),
                            Just (SomeNat (_    :: Proxy strideRows)), Just (SomeNat  (_    :: Proxy strideCols))) ->
                              let p1 = natDict pkr
                                  p2 = natDict pkc
                                  p3 = natDict chan
                              in  case ( p1 %* p2 %* p3
                                        , natDict pinr %* natDict chan
                                        -- Fake it till you make it.
                                        , (unsafeCoerce (Dict :: Dict ()) :: Dict ( ( 'D3 outRows outCols filters) ~ h ))
                                        , (unsafeCoerce (Dict :: Dict ()) :: Dict (((outRows - 1) * strideRows) ~ (inRows - kernelRows)))
                                        , (unsafeCoerce (Dict :: Dict ()) :: Dict (((outCols - 1) * strideCols) ~ (inCols - kernelCols)))) of
                                    (Dict, Dict, Dict, Dict, Dict) -> do
                                        conv <- genConvolution
                                        pure (SomeNetwork (conv :~> rest :: Network ( Convolution channels filters kernelRows kernelCols strideRows strideCols ': layers ) ( ('D3 inRows inCols channels) ': h ': hs )))
                          _ -> Gen.discard -- Can't occur
                   , do -- Build a Pooling layer
                        let output_r = natVal r
                        let output_c = natVal c
                        let output_f = natVal f

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

                        guard (input_r < 100)
                        guard (input_c < 100)

                        -- Remake types for input
                        case ( someNatVal output_f, someNatVal input_r, someNatVal input_c, someNatVal output_r, someNatVal output_c, someNatVal kernel_r, someNatVal kernel_c, someNatVal stride_r, someNatVal stride_c ) of
                          ( Just (SomeNat (chan :: Proxy filters)),
                            Just (SomeNat (pinr :: Proxy inRows)),     Just (SomeNat  (_    :: Proxy inCols)),
                            Just (SomeNat (_    :: Proxy outRows)),    Just (SomeNat  (_    :: Proxy outCols)),
                            Just (SomeNat (pkr  :: Proxy kernelRows)), Just (SomeNat  (pkc  :: Proxy kernelCols)),
                            Just (SomeNat (_    :: Proxy strideRows)), Just (SomeNat  (_    :: Proxy strideCols))) ->
                              let p1 = natDict pkr
                                  p2 = natDict pkc
                                  p3 = natDict chan
                              in  case ( p1 %* p2 %* p3
                                        , natDict pinr %* natDict chan
                                        -- Fake it till you make it.
                                        , (unsafeCoerce (Dict :: Dict ()) :: Dict ( ( 'D3 outRows outCols filters) ~ h ))
                                        , (unsafeCoerce (Dict :: Dict ()) :: Dict (((outRows - 1) * strideRows) ~ (inRows - kernelRows)))
                                        , (unsafeCoerce (Dict :: Dict ()) :: Dict (((outCols - 1) * strideCols) ~ (inCols - kernelCols)))) of
                                    (Dict, Dict, Dict, Dict, Dict) ->
                                        pure (SomeNetwork (Pooling :~> rest :: Network ( Pooling kernelRows kernelCols strideRows strideCols ': layers ) ( ('D3 inRows inCols filters) ': h ': hs )))
                          _ -> Gen.discard -- Can't occur
                   ]
    ]

-- | Test a partial derivative numerically for a random network and input
--
-- This is the most important test.
prop_auto_diff :: Property
prop_auto_diff = withDiscards 1000 . withTests 10000 . property $ do
  SomeNetwork (network :: Network layers shapes) <- forAll genNetwork
  (input  :: S (Head shapes))     <- forAllRender nice genOfShape
  (target :: S (Last shapes))     <- forAllRender nice oneUp
  (tested :: S (Head shapes))     <- forAllRender nice oneUp

  let (!tapes, !output) = runNetwork network input
  let (_, !backgrad)    = runGradient network tapes target
  let inputDiff         = input + tested * 0.00001
  let expected          = maxVal ( backgrad * tested )
  let (_, !outputDiff)  = runNetwork network inputDiff
  let result            = maxVal ( outputDiff * target - output * target ) / 0.00001

  result ~~~ expected

-- Make a shape where all are 0 except for 1 value, which is 1.
oneUp :: forall shape m. ( Monad m, SingI shape ) => Gen.Gen m (S shape)
oneUp =
  case ( sing :: Sing shape ) of
    D1Sing SNat ->
      let x = 0 :: S ( shape )
      in  case x of
              ( S1D x' ) -> do
              let ex = extract x'
              let len = VS.length ex
              ix <- choose 0 (len - 1)
              let nx = runST $ do ex' <- VS.thaw ex
                                  VS.write ex' ix 1.0
                                  VS.freeze ex'
              maybe Gen.discard pure . fromStorable $ nx

    D2Sing SNat SNat ->
      let x = 0 :: S ( shape )
      in  case x of
              ( S2D x' ) -> do
              let ex = flatten ( extract x' )
              let len = VS.length ex
              ix <- choose 0 (len - 1)
              let nx = runST $ do ex' <- VS.thaw ex
                                  VS.write ex' ix 1
                                  VS.freeze ex'
              maybe Gen.discard pure . fromStorable $ nx

    D3Sing SNat SNat SNat ->
      let x = 0 :: S ( shape )
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
  if abs (x - y) < 2e-5 then
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
tests = checkParallel $$(discover)
