{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE GADTs             #-}
{-# LANGUAGE ScopedTypeVariables             #-}
{-# LANGUAGE KindSignatures             #-}
{-# LANGUAGE ConstraintKinds         #-}
{-# LANGUAGE TypeOperators         #-}

{-# OPTIONS_GHC -fno-warn-missing-signatures #-}
module Test.Grenade.Layers.Convolution where

import           Unsafe.Coerce
import           Data.Constraint
import           Data.Proxy
import           Data.Singletons ()
import           GHC.TypeLits
import           GHC.TypeLits.Witnesses

import           Grenade.Core.Shape
import           Grenade.Core.Network
import           Grenade.Layers.Convolution

import qualified Numeric.LinearAlgebra.Static as HStatic

import           Disorder.Jack

import           Test.Jack.Hmatrix

data OpaqueConvolution :: * where
     OpaqueConvolution :: Convolution channels filters kernelRows kernelColumns strideRows strideColumns -> OpaqueConvolution

instance Show OpaqueConvolution where
    show (OpaqueConvolution n) = show n

genOpaqueOpaqueConvolution :: Jack OpaqueConvolution
genOpaqueOpaqueConvolution = do
    Just channels <- someNatVal <$> choose (1, 10)
    Just filters  <- someNatVal <$> choose (1, 10)
    Just kernel_h <- someNatVal <$> choose (2, 20)
    Just kernel_w <- someNatVal <$> choose (2, 20)
    Just stride_h <- someNatVal <$> choose (1, 10)
    Just stride_w <- someNatVal <$> choose (1, 10)
    case (channels, filters, kernel_h, kernel_w, stride_h, stride_w) of
       ( SomeNat (pch :: Proxy ch), SomeNat  (_   :: Proxy fl),
         SomeNat (pkr :: Proxy kr), SomeNat  (pkc :: Proxy kc),
         SomeNat (_   :: Proxy sr), SomeNat  (_   :: Proxy sc)) ->
          let p1 = natDict pkr
              p2 = natDict pkc
              p3 = natDict pch
          in  case p1 %* p2 %* p3 of
            Dict -> OpaqueConvolution <$> (Convolution <$> uniformSample <*> uniformSample :: Jack (Convolution ch fl kr kc sr sc))

prop_conv_net_witness =
  gamble genOpaqueOpaqueConvolution $ \onet ->
    (case onet of
       (OpaqueConvolution ((Convolution _ _) :: Convolution channels filters kernelRows kernelCols strideRows strideCols)) -> True
    ) :: Bool

prop_conv_net =
  gamble genOpaqueOpaqueConvolution $ \onet ->
    (case onet of
       (OpaqueConvolution (convLayer@(Convolution _ _) :: Convolution channels filters kernelRows kernelCols strideRows strideCols)) ->
          let ok stride kernel = [extent | extent <- [(kernel + 1) .. 30 ], (extent - kernel) `mod` stride == 0]
              ch = fromIntegral $ natVal (Proxy :: Proxy channels)
              kr = fromIntegral $ natVal (Proxy :: Proxy kernelRows)
              kc = fromIntegral $ natVal (Proxy :: Proxy kernelCols)
              sr = fromIntegral $ natVal (Proxy :: Proxy strideRows)
              sc = fromIntegral $ natVal (Proxy :: Proxy strideCols)

          in  gamble (elements (ok sr kr)) $ \er ->
                  gamble (elements (ok sc kc)) $ \ec ->
                      let i  = fromIntegral (er * ec * ch)
                          rr = ((er - kr) `div` sr) + 1
                          rc = ((ec - kc) `div` sc) + 1
                          er' = someNatVal er
                          ec' = someNatVal ec
                          rr' = someNatVal rr
                          rc' = someNatVal rc
                      in gamble (vectorOf i sizedRealFrac) $ \(input :: [Double]) ->
                          case (er', ec', rr', rc') of
                            (Just (SomeNat (pinr :: Proxy inRows)), Just (SomeNat (_  :: Proxy inCols)), Just (SomeNat (pour :: Proxy outRows)), Just (SomeNat (_ :: Proxy outCols))) ->
                              let p1 = natDict pinr
                                  p2 = natDict pour
                              in  case (  p1 %* natDict (Proxy :: Proxy channels)
                                        , p2 %* natDict (Proxy :: Proxy filters)
                                        -- Fake it till you make it.
                                        , (unsafeCoerce (Dict :: Dict ()) :: Dict (((outRows - 1) * strideRows) ~ (inRows - kernelRows)))
                                        , (unsafeCoerce (Dict :: Dict ()) :: Dict (((outCols - 1) * strideCols) ~ (inCols - kernelCols)))) of
                                (Dict, Dict, Dict, Dict) -> let x :: S' ('D3 outRows outCols filters) = runForwards convLayer ((S3D' (HStatic.matrix input)) :: S' ('D3 inRows inCols channels))
                                                            in  x `seq` True
                            _ -> False
    ) :: Property

return []
tests :: IO Bool
tests = $quickCheckAll
