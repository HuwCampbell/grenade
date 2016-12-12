{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE GADTs             #-}
{-# OPTIONS_GHC -fno-warn-missing-signatures #-}
module Test.Grenade.Layers.Convolution where

import           Grenade.Core.Shape
import           Grenade.Core.Vector as Grenade
import           Grenade.Core.Network
import           Grenade.Layers.Convolution
import           Grenade.Layers.Internal.Convolution

import           Numeric.LinearAlgebra hiding (uniformSample, konst, (===))
import qualified Numeric.LinearAlgebra.Static as HStatic

import           Test.QuickCheck hiding ((><))

prop_im2col_no_stride = once $
 let input = (3><4)
               [ 1.0,  2.0,  3.0,  4.0
               , 5.0,  6.0,  7.0,  8.0
               , 9.0, 10.0, 11.0, 12.0 ]
     expected = (6><4)
               [ 1.0,  2.0,  5.0,  6.0
               , 2.0,  3.0,  6.0,  7.0
               , 3.0,  4.0,  7.0,  8.0
               , 5.0,  6.0,  9.0,  10.0
               , 6.0,  7.0,  10.0, 11.0
               , 7.0,  8.0,  11.0, 12.0 ]
     out = im2colUnsafe 2 2 1 1 input
 in expected === out

prop_im2col_c = once $
 let input = (3><4)
               [ 1.0,  2.0,  3.0,  4.0
               , 5.0,  6.0,  7.0,  8.0
               , 9.0, 10.0, 11.0, 12.0 ]
     expected = (6><4)
               [ 1.0,  2.0,  5.0,  6.0
               , 2.0,  3.0,  6.0,  7.0
               , 3.0,  4.0,  7.0,  8.0
               , 5.0,  6.0,  9.0,  10.0
               , 6.0,  7.0,  10.0, 11.0
               , 7.0,  8.0,  11.0, 12.0 ]
     out = im2col_c 2 2 1 1 input
 in expected === out

prop_im2col_stride = once $
 let input = (3><4)
               [ 1.0,  2.0,  3.0,  4.0
               , 5.0,  6.0,  7.0,  8.0
               , 9.0, 10.0, 11.0, 12.0 ]
     expected = (4><4)
               [ 1.0,  2.0,  5.0,  6.0
               , 3.0,  4.0,  7.0,  8.0
               , 5.0,  6.0,  9.0,  10.0
               , 7.0,  8.0,  11.0, 12.0 ]
     out = im2colUnsafe 2 2 1 2 input
 in expected === out

prop_im2col_c_stride = once $
 let input = (3><4)
               [ 1.0,  2.0,  3.0,  4.0
               , 5.0,  6.0,  7.0,  8.0
               , 9.0, 10.0, 11.0, 12.0 ]
     expected = (4><4)
               [ 1.0,  2.0,  5.0,  6.0
               , 3.0,  4.0,  7.0,  8.0
               , 5.0,  6.0,  9.0,  10.0
               , 7.0,  8.0,  11.0, 12.0 ]
     out = im2col_c 2 2 1 2 input
 in expected === out

prop_im2col_other = once $
 let input = (3><4)
               [ 1.0,  2.0,  3.0,  4.0
               , 5.0,  6.0,  7.0,  8.0
               , 9.0, 10.0, 11.0, 12.0 ]
     expected = (2><6)
               [ 1.0,  2.0,  5.0,  6.0 , 9.0,  10.0
               , 3.0,  4.0,  7.0,  8.0 , 11.0 ,12.0 ]
     out = im2colUnsafe 3 2 1 2 input
 in expected === out

prop_im2col_c_other = once $
 let input = (3><4)
               [ 1.0,  2.0,  3.0,  4.0
               , 5.0,  6.0,  7.0,  8.0
               , 9.0, 10.0, 11.0, 12.0 ]
     expected = (2><6)
               [ 1.0,  2.0,  5.0,  6.0 , 9.0,  10.0
               , 3.0,  4.0,  7.0,  8.0 , 11.0 ,12.0 ]
     out = im2col_c 3 2 1 2 input
 in expected === out

prop_im2col_bigger = once $
 let input = (7><7) [ 1.0 .. ]
 in im2colUnsafe 5 5 2 2 input === im2col_c 5 5 2 2 input

-- If there's no overlap (stride is the same size as the kernel)
-- then col2im . im2col should be symmetric.
prop_im2col_sym_on_same_stride = once $
 let input = (3><4)
               [ 1.0,  2.0,  3.0,  4.0
               , 5.0,  6.0,  7.0,  8.0
               , 9.0, 10.0, 11.0, 12.0 ]
     out = col2imUnsafe 3 2 3 2 3 4 . im2colUnsafe 3 2 3 2 $ input
 in input === out

-- If there's no overlap (stride is the same size as the kernel)
-- then col2im . im2col should be symmetric.
prop_im2colunsafe_sym_on_same_stride = once $
 let input = (3><4)
               [ 1.0,  2.0,  3.0,  4.0
               , 5.0,  6.0,  7.0,  8.0
               , 9.0, 10.0, 11.0, 12.0 ]
     out = col2imUnsafe 3 2 3 2 3 4 . im2colUnsafe 3 2 3 2 $ input
 in input === out


-- If there is an overlap, then the gradient passed back should be
-- the sum of the gradients across the filters.
prop_im2col_col2im_additive = once $
 let input = (3><4)
               [ 1.0,  1.0,  1.0,  1.0
               , 1.0,  1.0,  1.0,  1.0
               , 1.0,  1.0,  1.0,  1.0 ]
     expected = (3><4)
               [ 1.0,  2.0,  2.0,  1.0
               , 2.0,  4.0,  4.0,  2.0
               , 1.0,  2.0,  2.0,  1.0 ]
     out = col2imUnsafe 2 2 1 1 3 4 . im2colUnsafe 2 2 1 1 $ input
 in expected === out

prop_simple_conv_forwards = once $
  -- Create a convolution kernel with 4 filters.
  -- [ 1, 0    [ 0, 1    [ 0, 1    [ 0, 0
  -- , 0,-1 ]  ,-1, 0 ]  , 1, 0 ]  ,-1,-1 ]
  let myKernel = HStatic.matrix
                 [ 1.0,  0.0,  0.0,  0.0
                 , 0.0,  1.0,  1.0,  0.0
                 , 0.0, -1.0,  1.0, -1.0
                 ,-1.0,  0.0,  0.0, -1.0 ] :: HStatic.L 4 4
      zeroKernel = HStatic.matrix
                 [ 0.0,  0.0,  0.0,  0.0
                 , 0.0,  0.0,  0.0,  0.0
                 , 0.0,  0.0,  0.0,  0.0
                 , 0.0,  0.0,  0.0,  0.0 ] :: HStatic.L 4 4

      expectedGradient = HStatic.matrix
                 [ 1.0, 0.0, 0.0, 2.0
                 , 2.0, 0.0, 0.0, 5.0
                 , 3.0, 0.0, 0.0, 4.0
                 , 4.0, 0.0, 0.0, 6.0 ] :: HStatic.L 4 4

      convLayer = Convolution myKernel zeroKernel :: Convolution 1 4 2 2 1 1

      input = S2D' (HStatic.matrix
                 [ 1.0, 2.0, 5.0
                 , 3.0, 4.0, 6.0] :: HStatic.L 2 3)

      expect = [ HStatic.matrix
                 [ -3.0 , -4.0  ] :: HStatic.L 1 2
               , HStatic.matrix
                 [ -1.0 ,  1.0  ] :: HStatic.L 1 2
               , HStatic.matrix
                 [  5.0 ,  9.0  ] :: HStatic.L 1 2
               , HStatic.matrix
                 [ -7.0 , -10.0 ] :: HStatic.L 1 2] :: [HStatic.L 1 2]
      out  = runForwards convLayer input :: S' ('D3 1 2 4)

      grad =  S3D' ( mkVector
               [ HStatic.matrix
                 [ 1 , 0 ] :: HStatic.L 1 2
               , HStatic.matrix
                 [ 0 , 0 ] :: HStatic.L 1 2
               , HStatic.matrix
                 [ 0 , 0 ] :: HStatic.L 1 2
               , HStatic.matrix
                 [ 0 , 1 ] :: HStatic.L 1 2] ) :: S' ('D3 1 2 4)

      expectBack = (HStatic.matrix
                   [  1.0,  0.0, 0.0
                   ,  0.0, -2.0,-1.0] :: HStatic.L 2 3)
      (nc, inX)  =  runBackwards convLayer input grad

  in case (out, inX, nc) of
    (S3D' out' , S2D' inX', Convolution' backGrad)
      -> ((HStatic.extract <$> expect) === (HStatic.extract <$> vecToList out'))
      .&&. (HStatic.extract expectBack === HStatic.extract inX')
      .&&. (HStatic.extract expectedGradient === HStatic.extract backGrad)

prop_vid2col_no_stride = once $
 let input = [(3><4)
               [ 1.0,  2.0,  3.0,  4.0
               , 5.0,  6.0,  7.0,  8.0
               , 9.0, 10.0, 11.0, 12.0 ]
             , (3><4)
               [ 21.0,  22.0,  23.0,  24.0
               , 25.0,  26.0,  27.0,  28.0
               , 29.0,  30.0,  31.0,  32.0 ] ]
     expected = (6><8)
               [ 1.0,  2.0,  5.0,  6.0  , 21.0,  22.0,  25.0,  26.0
               , 2.0,  3.0,  6.0,  7.0  , 22.0,  23.0,  26.0,  27.0
               , 3.0,  4.0,  7.0,  8.0  , 23.0,  24.0,  27.0,  28.0
               , 5.0,  6.0,  9.0,  10.0 , 25.0,  26.0,  29.0,  30.0
               , 6.0,  7.0,  10.0, 11.0 , 26.0,  27.0,  30.0,  31.0
               , 7.0,  8.0,  11.0, 12.0 , 27.0,  28.0,  31.0,  32.0 ]
     out = vid2colUnsafe 2 2 1 1 3 4 input
 in expected === out

prop_vid2col_stride = once $
 let input = [(3><4)
               [ 1.0,  2.0,  3.0,  4.0
               , 5.0,  6.0,  7.0,  8.0
               , 9.0, 10.0, 11.0, 12.0 ]
             , (3><4)
               [ 21.0,  22.0,  23.0,  24.0
               , 25.0,  26.0,  27.0,  28.0
               , 29.0,  30.0,  31.0,  32.0 ] ]
     expected = (4><8)
               [ 1.0,  2.0,  5.0,  6.0  , 21.0, 22.0, 25.0, 26.0
               , 3.0,  4.0,  7.0,  8.0  , 23.0, 24.0, 27.0, 28.0
               , 5.0,  6.0,  9.0,  10.0 , 25.0, 26.0, 29.0, 30.0
               , 7.0,  8.0,  11.0, 12.0 , 27.0, 28.0, 31.0, 32.0 ]
     out = vid2colUnsafe 2 2 1 2 3 4 input
 in expected === out

prop_vid2col_invert = once $
 let input = [(3><4)
               [ 1.0,  2.0,  3.0,  4.0
               , 5.0,  6.0,  7.0,  8.0
               , 9.0, 10.0, 11.0, 12.0 ]
             , (3><4)
               [ 21.0,  22.0,  23.0,  24.0
               , 25.0,  26.0,  27.0,  28.0
               , 29.0,  30.0,  31.0,  32.0 ] ]
     out = col2vidUnsafe 3 2 3 2 3 4 . vid2colUnsafe 3 2 3 2 3 4 $ input
 in input === out


-- This test show that 2D convs act the same
-- 3D convs with one layer
prop_single_conv_forwards = once $
  -- Create a convolution kernel with 4 filters.
  -- [ 1, 0    [ 0, 1    [ 0, 1    [ 0, 0
  -- , 0,-1 ]  ,-1, 0 ]  , 1, 0 ]  ,-1,-1 ]
  let myKernel = (HStatic.matrix
                 [ 1.0,  0.0,  0.0,  0.0
                 , 0.0,  1.0,  1.0,  0.0
                 , 0.0, -1.0,  1.0, -1.0
                 ,-1.0,  0.0,  0.0, -1.0 ] :: HStatic.L 4 4)
      zeroKernel = (HStatic.matrix
                 [ 0.0,  0.0,  0.0,  0.0
                 , 0.0,  0.0,  0.0,  0.0
                 , 0.0,  0.0,  0.0,  0.0
                 , 0.0,  0.0,  0.0,  0.0 ] :: HStatic.L 4 4)

      expectedGradient = (HStatic.matrix
                 [ 1.0, 0.0, 0.0, 2.0
                 , 2.0, 0.0, 0.0, 5.0
                 , 3.0, 0.0, 0.0, 4.0
                 , 4.0, 0.0, 0.0, 6.0 ] :: HStatic.L 4 4)

      convLayer = Convolution myKernel zeroKernel :: Convolution 1 4 2 2 1 1

      input = S3D' ( mkVector [HStatic.matrix
                 [ 1.0, 2.0, 5.0
                 , 3.0, 4.0, 6.0] :: HStatic.L 2 3] ) :: S' ('D3 2 3 1)

      expect = [HStatic.matrix
                 [ -3.0 , -4.0  ] :: HStatic.L 1 2
               ,HStatic.matrix
                 [ -1.0 ,  1.0  ] :: HStatic.L 1 2
               ,HStatic.matrix
                 [  5.0 ,  9.0  ] :: HStatic.L 1 2
               ,HStatic.matrix
                 [ -7.0 , -10.0 ] :: HStatic.L 1 2] :: [HStatic.L 1 2]
      out  = runForwards convLayer input :: S' ('D3 1 2 4)

      grad =  S3D' ( mkVector
               [HStatic.matrix
                 [ 1 , 0 ] :: HStatic.L 1 2
               ,HStatic.matrix
                 [ 0 , 0 ] :: HStatic.L 1 2
               ,HStatic.matrix
                 [ 0 , 0 ] :: HStatic.L 1 2
               ,HStatic.matrix
                 [ 0 , 1 ] :: HStatic.L 1 2] ) :: S' ('D3 1 2 4)

      expectBack = (HStatic.matrix
                   [  1.0,  0.0, 0.0
                   ,  0.0, -2.0,-1.0] :: HStatic.L 2 3)
      (nc, inX)  = runBackwards convLayer input grad

  in case (out, inX, nc) of
    (S3D' out' , S3D' inX', Convolution' backGrad)
      ->   ((HStatic.extract <$> expect)  === (HStatic.extract <$> vecToList out'))
      .&&. ([HStatic.extract expectBack]  === (HStatic.extract <$> vecToList inX'))
      .&&. (HStatic.extract expectedGradient === HStatic.extract backGrad)

return []
tests :: IO Bool
tests = $quickCheckAll
