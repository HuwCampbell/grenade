{-# LANGUAGE CPP                   #-}
{-# LANGUAGE TemplateHaskell       #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE KindSignatures        #-}
{-# LANGUAGE ConstraintKinds       #-}
{-# LANGUAGE TypeOperators         #-}
{-# OPTIONS_GHC -fno-warn-missing-signatures #-}
module Test.Grenade.Layers.Convolution where

import           Unsafe.Coerce
import           Data.Constraint
import           Data.Proxy
import           Data.Singletons ()
import           GHC.TypeLits
import           GHC.TypeLits.Witnesses

#if MIN_VERSION_base(4,9,0)
import           Data.Kind (Type)
#endif

import           Grenade.Core
import           Grenade.Layers.Convolution

import           Hedgehog
import qualified Hedgehog.Gen as Gen

import           Test.Hedgehog.Hmatrix
import           Test.Hedgehog.TypeLits
import           Test.Hedgehog.Compat

data OpaqueConvolution :: Type where
     OpaqueConvolution :: Convolution channels filters kernelRows kernelColumns strideRows strideColumns -> OpaqueConvolution

instance Show OpaqueConvolution where
    show (OpaqueConvolution n) = show n

genConvolution :: ( KnownNat channels
                  , KnownNat filters
                  , KnownNat kernelRows
                  , KnownNat kernelColumns
                  , KnownNat strideRows
                  , KnownNat strideColumns
                  , KnownNat kernelFlattened
                  , kernelFlattened ~ (kernelRows * kernelColumns * channels)
                  ) => Gen (Convolution channels filters kernelRows kernelColumns strideRows strideColumns)
genConvolution = Convolution <$> uniformSample <*> uniformSample

genOpaqueOpaqueConvolution :: Gen OpaqueConvolution
genOpaqueOpaqueConvolution = do
    channels <- genNat
    filters  <- genNat
    kernel_h <- genNat
    kernel_w <- genNat
    stride_h <- genNat
    stride_w <- genNat
    case (channels, filters, kernel_h, kernel_w, stride_h, stride_w) of
       ( SomeNat (pch :: Proxy ch), SomeNat  (_   :: Proxy fl),
         SomeNat (pkr :: Proxy kr), SomeNat  (pkc :: Proxy kc),
         SomeNat (_   :: Proxy sr), SomeNat  (_   :: Proxy sc)) ->
          let p1 = natDict pkr
              p2 = natDict pkc
              p3 = natDict pch
          in  case p1 %* p2 %* p3 of
            Dict -> OpaqueConvolution <$> (genConvolution :: Gen (Convolution ch fl kr kc sr sc))

prop_conv_net_witness = property $
  blindForAll genOpaqueOpaqueConvolution >>= \onet ->
    case onet of
       (OpaqueConvolution ((Convolution _ _) :: Convolution channels filters kernelRows kernelCols strideRows strideCols)) -> success


prop_conv_net = property $
  blindForAll genOpaqueOpaqueConvolution >>= \onet ->
    case onet of
       (OpaqueConvolution (convLayer@(Convolution _ _) :: Convolution channels filters kernelRows kernelCols strideRows strideCols)) ->
          let ok stride kernel = [extent | extent <- [(kernel + 1) .. 30 ], (extent - kernel) `mod` stride == 0]
              kr = fromIntegral $ natVal (Proxy :: Proxy kernelRows)
              kc = fromIntegral $ natVal (Proxy :: Proxy kernelCols)
              sr = fromIntegral $ natVal (Proxy :: Proxy strideRows)
              sc = fromIntegral $ natVal (Proxy :: Proxy strideCols)

          in  forAll (Gen.element (ok sr kr)) >>= \er ->
                  forAll (Gen.element (ok sc kc)) >>= \ec ->
                      let rr = ((er - kr) `div` sr) + 1
                          rc = ((ec - kc) `div` sc) + 1
                          Just er' = someNatVal er
                          Just ec' = someNatVal ec
                          Just rr' = someNatVal rr
                          Just rc' = someNatVal rc
                      in case (er', ec', rr', rc') of
                            ( SomeNat (pinr :: Proxy inRows), SomeNat (_  :: Proxy inCols), SomeNat (pour :: Proxy outRows), SomeNat (_ :: Proxy outCols)) ->
                              case ( natDict pinr %* natDict (Proxy :: Proxy channels)
                                   , natDict pour %* natDict (Proxy :: Proxy filters)
                                   -- Fake it till you make it.
                                   , (unsafeCoerce (Dict :: Dict ()) :: Dict (((outRows - 1) * strideRows) ~ (inRows - kernelRows)))
                                   , (unsafeCoerce (Dict :: Dict ()) :: Dict (((outCols - 1) * strideCols) ~ (inCols - kernelCols)))) of
                                (Dict, Dict, Dict, Dict) ->
                                    blindForAll (S3D <$> uniformSample) >>= \(input :: S ('D3 inRows inCols channels)) ->
                                        let (tape, output :: S ('D3 outRows outCols filters)) = runForwards convLayer input
                                            backed :: (Gradient (Convolution channels filters kernelRows kernelCols strideRows strideCols), S ('D3 inRows inCols channels))
                                                                                              = runBackwards convLayer tape output
                                        in  backed `seq` success


tests :: IO Bool
tests = checkParallel $$(discover)
