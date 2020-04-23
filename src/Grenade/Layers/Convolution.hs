{-# LANGUAGE CPP                   #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE UndecidableInstances  #-}
{-|
Module      : Grenade.Layers.Convolution
Description : Convolution layer
Copyright   : (c) Huw Campbell, 2016-2017
License     : BSD2
Stability   : experimental

This module provides the Convolution layer, which is critical in many computer vision tasks.

-}
module Grenade.Layers.Convolution (
    Convolution (..)
  , Convolution' (..)
  , SpecConvolution (..)
  , specConvolution2DInput
  , specConvolution3DInput
  , convolution
  ) where

import           Control.DeepSeq                     (NFData (..))
import           Control.Monad.Primitive             (PrimBase, PrimState)
import           Data.Constraint                     (Dict (..))
import           Data.Kind                           (Type)
import           Data.Maybe
import           Data.Proxy
import           Data.Reflection                     (reifyNat)
import           Data.Serialize
import           Data.Singletons
import           Data.Singletons.Prelude.Num         ((%*))
import           Data.Singletons.TypeLits            hiding (natVal)
import           GHC.TypeLits
import           Numeric.LinearAlgebra               hiding (konst, uniformSample)
import qualified Numeric.LinearAlgebra               as LA
import           Numeric.LinearAlgebra.Static        hiding (build, toRows, (|||))
import           System.Random.MWC                   (Gen)
import           Unsafe.Coerce                       (unsafeCoerce)

import           Grenade.Core
import           Grenade.Dynamic
import           Grenade.Dynamic.Internal.Build
import           Grenade.Layers.Internal.Convolution
import           Grenade.Layers.Internal.Update
import           Grenade.Utils.ListStore


-- | A convolution layer for a neural network.
--   This uses the im2col convolution trick popularised by Caffe, which essentially turns the
--   many, many, many, many loop convolution into a single matrix multiplication.
--
--   The convolution layer takes all of the kernels for the convolution, which are flattened
--   and then put into columns in the matrix.
--
--   The kernel size dictates which input and output sizes will "fit". Fitting the equation:
--   `out = (in - kernel) / stride + 1` for both dimensions.
--
--   One probably shouldn't build their own layer, but rather use the randomConvolution function.
data Convolution :: Nat -- Number of channels, for the first layer this could be RGB for instance.
                 -> Nat -- Number of filters, this is the number of channels output by the layer.
                 -> Nat -- The number of rows in the kernel filter
                 -> Nat -- The number of column in the kernel filter
                 -> Nat -- The row stride of the convolution filter
                 -> Nat -- The columns stride of the convolution filter
                 -> Type where
  Convolution :: ( KnownNat channels
                 , KnownNat filters
                 , KnownNat kernelRows
                 , KnownNat kernelColumns
                 , KnownNat strideRows
                 , KnownNat strideColumns
                 , KnownNat kernelFlattened
                 , kernelFlattened ~ (kernelRows * kernelColumns * channels))
              => !(L kernelFlattened filters) -- The kernel filter weights
              -> !(ListStore (L kernelFlattened filters)) -- The last kernel update (or momentum)
              -> Convolution channels filters kernelRows kernelColumns strideRows strideColumns

instance NFData (Convolution channels filters kernelRows kernelColumns strideRows strideColumns) where
  rnf (Convolution a b) = rnf a `seq` rnf b


data Convolution' :: Nat -- Number of channels, for the first layer this could be RGB for instance.
                  -> Nat -- Number of filters, this is the number of channels output by the layer.
                  -> Nat -- The number of rows in the kernel filter
                  -> Nat -- The number of column in the kernel filter
                  -> Nat -- The row stride of the convolution filter
                  -> Nat -- The columns stride of the convolution filter
                  -> Type where
  Convolution' :: ( KnownNat channels
                  , KnownNat filters
                  , KnownNat kernelRows
                  , KnownNat kernelColumns
                  , KnownNat strideRows
                  , KnownNat strideColumns
                  , KnownNat kernelFlattened
                  , kernelFlattened ~ (kernelRows * kernelColumns * channels))
               => !(L kernelFlattened filters) -- The kernel filter gradient
               -> Convolution' channels filters kernelRows kernelColumns strideRows strideColumns

instance NFData (Convolution' c f k k' s s') where
  rnf (Convolution' a) = rnf a


instance Show (Convolution c f k k' s s') where
  show (Convolution a _) = renderConv a
    where
      renderConv mm =
        let m = extract mm
            ky = fromIntegral $ natVal (Proxy :: Proxy k)
            rs = LA.toColumns m
            ms = map (take ky) $ toLists . reshape ky <$> rs
            render n'
              | n' <= 0.2 = ' '
              | n' <= 0.4 = '.'
              | n' <= 0.6 = '-'
              | n' <= 0.8 = '='
              | otherwise = '#'
            px = (fmap . fmap . fmap) render ms
         in unlines $ foldl1 (zipWith (\a' b' -> a' ++ "   |   " ++ b')) px


instance ( KnownNat channels
         , KnownNat filters
         , KnownNat kernelRows
         , KnownNat kernelColumns
         , KnownNat strideRows
         , KnownNat strideColumns
         , KnownNat ((kernelRows * kernelColumns) * channels)
         , KnownNat (filters * ((kernelRows * kernelColumns) * channels))
         ) =>
         RandomLayer (Convolution channels filters kernelRows kernelColumns strideRows strideColumns) where
  createRandomWith m gen = do
    wN <- getRandomMatrix i i m gen
    return $ Convolution wN mkListStore
    where
      i = natVal (Proxy :: Proxy ((kernelRows * kernelColumns) * channels))


instance ( KnownNat channels
         , KnownNat filters
         , KnownNat kernelRows
         , KnownNat kernelColumns
         , KnownNat strideRows
         , KnownNat strideColumns
         , KnownNat (kernelRows * kernelColumns * channels)
         , kernelFlattened ~ (kernelRows * kernelColumns * channels)
         ) => UpdateLayer (Convolution channels filters kernelRows kernelColumns strideRows strideColumns) where
  type Gradient (Convolution channels filters kernelRows kernelColumns strideRows strideColumns) = (Convolution' channels filters kernelRows kernelColumns strideRows strideColumns)

  type MomentumStore (Convolution channels filters kernelRows kernelColumns strideRows strideColumns)  = ListStore (L (kernelRows * kernelColumns * channels) filters)

  runUpdate opt@OptSGD{} x@(Convolution oldKernel store) (Convolution' kernelGradient) =
    let  momentum = getData opt x store
         result = descendMatrix opt (MatrixValuesSGD oldKernel kernelGradient momentum)
         newStore = setData opt x store (matrixMomentum result)
    in Convolution (matrixActivations result) newStore
  runUpdate opt@OptAdam{} x@(Convolution oldKernel store) (Convolution' kernelGradient) =
    let (m, v) = toTuple $ getData opt x store
        result = descendMatrix opt (MatrixValuesAdam (getStep store) oldKernel kernelGradient m v)
        newStore = setData opt x store [matrixM result, matrixV result]
    in Convolution (matrixActivations result) newStore
    where toTuple [m ,v] = (m, v)
          toTuple xs = error $ "unexpected input of length " ++ show (length xs) ++ "in toTuple in Convolution.hs"

instance ( KnownNat channels
         , KnownNat filters
         , KnownNat kernelRows
         , KnownNat kernelColumns
         , KnownNat strideRows
         , KnownNat strideColumns
         , KnownNat (kernelRows * kernelColumns * channels)
         ) =>
         LayerOptimizerData (Convolution channels filters kernelRows kernelColumns strideRows strideColumns) (Optimizer 'SGD) where
  type MomentumExpOptResult (Convolution channels filters kernelRows kernelColumns strideRows strideColumns) (Optimizer 'SGD) = L (kernelRows * kernelColumns * channels) filters
  type MomentumDataType (Convolution channels filters kernelRows kernelColumns strideRows strideColumns) (Optimizer 'SGD) = L (kernelRows * kernelColumns * channels) filters
  getData opt x store = head $ getListStore opt x store
  setData opt x store = setListStore opt x store . return
  newData _ _ = konst 0

instance ( KnownNat channels
         , KnownNat filters
         , KnownNat kernelRows
         , KnownNat kernelColumns
         , KnownNat strideRows
         , KnownNat strideColumns
         , KnownNat (kernelRows * kernelColumns * channels)
         ) => LayerOptimizerData (Convolution channels filters kernelRows kernelColumns strideRows strideColumns) (Optimizer 'Adam) where
  type MomentumExpOptResult (Convolution channels filters kernelRows kernelColumns strideRows strideColumns) (Optimizer 'Adam)  = [L (kernelRows * kernelColumns * channels) filters]
  type MomentumDataType (Convolution channels filters kernelRows kernelColumns strideRows strideColumns) (Optimizer 'Adam) = L (kernelRows * kernelColumns * channels) filters
  getData = getListStore
  setData = setListStore
  newData _ _ = konst 0


instance ( KnownNat channels
         , KnownNat filters
         , KnownNat kernelRows
         , KnownNat kernelColumns
         , KnownNat strideRows
         , KnownNat strideColumns
         , KnownNat (kernelRows * kernelColumns * channels)
         ) => Serialize (Convolution channels filters kernelRows kernelColumns strideRows strideColumns) where
  put (Convolution w store) = do
    putListOf put . toList . flatten . extract $ w
    put (fmap (toList . flatten . extract) store)
  get = do
      let f  = fromIntegral $ natVal (Proxy :: Proxy filters)
      wN    <- maybe (fail "Vector of incorrect size") return . create . reshape f . LA.fromList =<< getListOf get
      store <- fmap (fromMaybe (error "Vector of incorrect size") . create . reshape f . LA.fromList)  <$> get
      return $ Convolution wN store

instance ( KnownNat channels
         , KnownNat filters
         , KnownNat kernelRows
         , KnownNat kernelColumns
         , KnownNat strideRows
         , KnownNat strideColumns
         , KnownNat (kernelRows * kernelColumns * channels)
         ) =>
         Serialize (Convolution' channels filters kernelRows kernelColumns strideRows strideColumns) where
  put (Convolution' w) = putListOf put . toList . flatten . extract $ w
  get = do
    let f = fromIntegral $ natVal (Proxy :: Proxy channels)
    wN <- maybe (fail "Vector of incorrect size") return . create . reshape f . LA.fromList =<< getListOf get
    return $ Convolution' wN

-- | A three dimensional image (or 2d with many channels) can have
--   an appropriately sized convolution filter run across it.
instance ( KnownNat kernelRows
         , KnownNat kernelCols
         , KnownNat filters
         , KnownNat strideRows
         , KnownNat strideCols
         , KnownNat inputRows
         , KnownNat inputCols
         , KnownNat outputRows
         , KnownNat outputCols
         , KnownNat channels
         , ((outputRows - 1) * strideRows) ~ (inputRows - kernelRows)
         , ((outputCols - 1) * strideCols) ~ (inputCols - kernelCols)
         , KnownNat (kernelRows * kernelCols * channels)
         , KnownNat (outputRows * filters)
         ) => Layer (Convolution channels filters kernelRows kernelCols strideRows strideCols) ('D3 inputRows inputCols channels) ('D3 outputRows outputCols filters) where

  type Tape (Convolution channels filters kernelRows kernelCols strideRows strideCols) ('D3 inputRows inputCols channels) ('D3 outputRows outputCols filters) = S ('D3 inputRows inputCols channels)

  runForwards (Convolution kernel _) (S3D input) =
    let ex = extract input
        ek = extract kernel
        ix = fromIntegral $ natVal (Proxy :: Proxy inputRows)
        iy = fromIntegral $ natVal (Proxy :: Proxy inputCols)
        kx = fromIntegral $ natVal (Proxy :: Proxy kernelRows)
        ky = fromIntegral $ natVal (Proxy :: Proxy kernelCols)
        sx = fromIntegral $ natVal (Proxy :: Proxy strideRows)
        sy = fromIntegral $ natVal (Proxy :: Proxy strideCols)
        ox = fromIntegral $ natVal (Proxy :: Proxy outputRows)
        oy = fromIntegral $ natVal (Proxy :: Proxy outputCols)

        c  = vid2col kx ky sx sy ix iy ex
        mt = c LA.<> ek
        r  = col2vid 1 1 1 1 ox oy mt
        rs = fromJust . create $ r
    in  (S3D input, S3D rs)
  runBackwards (Convolution kernel _) (S3D input) (S3D dEdy) =
    let ex = extract input
        ix = fromIntegral $ natVal (Proxy :: Proxy inputRows)
        iy = fromIntegral $ natVal (Proxy :: Proxy inputCols)
        kx = fromIntegral $ natVal (Proxy :: Proxy kernelRows)
        ky = fromIntegral $ natVal (Proxy :: Proxy kernelCols)
        sx = fromIntegral $ natVal (Proxy :: Proxy strideRows)
        sy = fromIntegral $ natVal (Proxy :: Proxy strideCols)
        ox = fromIntegral $ natVal (Proxy :: Proxy outputRows)
        oy = fromIntegral $ natVal (Proxy :: Proxy outputCols)

        c  = vid2col kx ky sx sy ix iy ex

        eo = extract dEdy
        ek = extract kernel

        vs = vid2col 1 1 1 1 ox oy eo

        kN = fromJust . create $ tr c LA.<> vs

        dW = vs LA.<> tr ek

        xW = col2vid kx ky sx sy ix iy dW
    in  (Convolution' kN, S3D . fromJust . create $ xW)


-- | A two dimentional image may have a convolution filter applied to it
instance ( KnownNat kernelRows
         , KnownNat kernelCols
         , KnownNat filters
         , KnownNat strideRows
         , KnownNat strideCols
         , KnownNat inputRows
         , KnownNat inputCols
         , KnownNat outputRows
         , KnownNat outputCols
         , ((outputRows - 1) * strideRows) ~ (inputRows - kernelRows)
         , ((outputCols - 1) * strideCols) ~ (inputCols - kernelCols)
         , KnownNat (kernelRows * kernelCols * 1)
         , KnownNat (outputRows * filters)
         ) => Layer (Convolution 1 filters kernelRows kernelCols strideRows strideCols) ('D2 inputRows inputCols) ('D3 outputRows outputCols filters) where
  type Tape (Convolution 1 filters kernelRows kernelCols strideRows strideCols) ('D2 inputRows inputCols) ('D3 outputRows outputCols filters) = S ('D3 inputRows inputCols 1)
  runForwards c (S2D input) =
    runForwards c (S3D input :: S ('D3 inputRows inputCols 1))

  runBackwards c tape grads =
    case runBackwards c tape grads of
      (c', S3D back :: S ('D3 inputRows inputCols 1)) ->  (c', S2D back)


-- | A two dimensional image may have a convolution filter applied to it producing
--   a two dimensional image if both channels and filters is 1.
instance ( KnownNat kernelRows
         , KnownNat kernelCols
         , KnownNat strideRows
         , KnownNat strideCols
         , KnownNat inputRows
         , KnownNat inputCols
         , KnownNat outputCols
         , ((outputRows - 1) * strideRows) ~ (inputRows - kernelRows)
         , ((outputCols - 1) * strideCols) ~ (inputCols - kernelCols)
         , KnownNat (kernelRows * kernelCols * 1)
         , KnownNat (outputRows * 1)
         ) => Layer (Convolution 1 1 kernelRows kernelCols strideRows strideCols) ('D2 inputRows inputCols) ('D2 outputRows outputCols) where
  type Tape (Convolution 1 1 kernelRows kernelCols strideRows strideCols) ('D2 inputRows inputCols) ('D2 outputRows outputCols) = S ('D3 inputRows inputCols 1)
  runForwards c (S2D input) =
    case runForwards c (S3D input :: S ('D3 inputRows inputCols 1)) of
      (tps, S3D back :: S ('D3 outputRows outputCols 1)) ->  (tps, S2D back)

  runBackwards c tape (S2D grads) =
    case runBackwards c tape (S3D grads :: S ('D3 outputRows outputCols 1)) of
      (c', S3D back :: S ('D3 inputRows inputCols 1)) -> (c', S2D back)

-- | A three dimensional image can produce a 2D image from a convolution with 1 filter
instance ( KnownNat kernelRows
         , KnownNat kernelCols
         , KnownNat strideRows
         , KnownNat strideCols
         , KnownNat inputRows
         , KnownNat inputCols
         , KnownNat outputCols
         , KnownNat channels
         , ((outputRows - 1) * strideRows) ~ (inputRows - kernelRows)
         , ((outputCols - 1) * strideCols) ~ (inputCols - kernelCols)
         , KnownNat (kernelRows * kernelCols * channels)
         , KnownNat (outputRows * 1)
         ) => Layer (Convolution channels 1 kernelRows kernelCols strideRows strideCols) ('D3 inputRows inputCols channels) ('D2 outputRows outputCols) where
  type Tape (Convolution channels 1 kernelRows kernelCols strideRows strideCols) ('D3 inputRows inputCols channels) ('D2 outputRows outputCols) = S ('D3 inputRows inputCols channels)
  runForwards c input =
    case runForwards c input of
      (tps, S3D back :: S ('D3 outputRows outputCols 1)) ->  (tps, S2D back)

  runBackwards c tape (S2D grads) =
    runBackwards c tape (S3D grads :: S ('D3 outputRows outputCols 1))

-------------------- DynamicNetwork instance --------------------

instance (KnownNat channels, KnownNat filters, KnownNat kernelRows, KnownNat kernelColumns, KnownNat strideRows, KnownNat strideColumns) =>
         FromDynamicLayer (Convolution channels filters kernelRows kernelColumns strideRows strideColumns) where
  fromDynamicLayer inp _ _ =
    SpecNetLayer $
    SpecConvolution
      (tripleFromSomeShape inp)
      (natVal (Proxy :: Proxy channels))
      (natVal (Proxy :: Proxy filters))
      (natVal (Proxy :: Proxy kernelRows))
      (natVal (Proxy :: Proxy kernelColumns))
      (natVal (Proxy :: Proxy strideRows))
      (natVal (Proxy :: Proxy strideColumns))

instance ToDynamicLayer SpecConvolution where
  toDynamicLayer = toDynamicLayer'

toDynamicLayer' :: (PrimBase m) => WeightInitMethod -> Gen (PrimState m) -> SpecConvolution -> m SpecNetwork
toDynamicLayer' _ _ (SpecConvolution inp@(_, 1, 1) _ _ _ _ _ _) = error $ "1D input to a deconvolutional layer is not permited! you specified: " ++ show inp
toDynamicLayer' wInit gen (SpecConvolution (rows, cols, depth) ch fil kerRows kerCols strRows strCols) =
    reifyNat ch $ \(pxCh :: (KnownNat channels) => Proxy channels) ->
    reifyNat fil $ \(pxFil :: (KnownNat filters) => Proxy filters) ->
    reifyNat kerRows $ \(pxKerRows :: (KnownNat kernelRows) => Proxy kernelRows) ->
    reifyNat kerCols $ \(pxKerCols :: (KnownNat kernelCols) => Proxy kernelCols) ->
    reifyNat strRows $ \(_ :: (KnownNat strideRows) => Proxy strideRows) ->
    reifyNat strCols $ \(_ :: (KnownNat strideCols) => Proxy strideCols) ->
    reifyNat rows $ \(pxRows :: (KnownNat rows) => Proxy rows) ->
    reifyNat cols $ \(_ :: (KnownNat cols) => Proxy cols) ->
    reifyNat depth $ \(_ :: (KnownNat depth) => Proxy depth) ->
    reifyNat ((rows - 1) * strRows + kerRows) $ \(pxOutRows :: (KnownNat outRows) => Proxy outRows) ->
    reifyNat ((cols - 1) * strCols + kerCols) $ \(_ :: (KnownNat outCols) => Proxy outCols) ->
    case ((singByProxy pxKerRows %* singByProxy pxKerCols) %* singByProxy pxCh
         , singByProxy pxFil %* ((singByProxy pxKerRows %* singByProxy pxKerCols) %* singByProxy pxCh)
         , singByProxy pxOutRows %* singByProxy pxFil -- 'D3 representation
         , singByProxy pxRows %* singByProxy pxCh -- 'D3 representation
         ) of
      (SNat, SNat, SNat, SNat) | ch == 1 && fil == 1 ->
        case (unsafeCoerce (Dict :: Dict ()) :: Dict (channels ~ 1, filters ~ 1, ((outRows - 1) * strideRows) ~ (rows - kernelRows), ((outCols - 1) * strideCols) ~ (cols - kernelCols))) of
          Dict -> do
            (layer  :: Convolution 1 1 kernelRows kernelCols strideRows strideCols) <- createRandomWith wInit gen
            return $ SpecLayer layer (sing :: Sing ('D2 rows cols)) (sing :: Sing ('D2 outRows outCols))
      (SNat, SNat, SNat, SNat) | ch == 1 ->
        case (unsafeCoerce (Dict :: Dict ()) :: Dict (channels ~ 1, ((outRows - 1) * strideRows) ~ (rows - kernelRows), ((outCols - 1) * strideCols) ~ (cols - kernelCols))) of
          Dict -> do
            (layer  :: Convolution 1 filters kernelRows kernelCols strideRows strideCols) <- createRandomWith wInit gen
            return $ SpecLayer layer (sing :: Sing ('D2 rows cols)) (sing :: Sing ('D3 outRows outCols filters))
      (SNat, SNat, SNat, SNat) | fil == 1 ->
        case (unsafeCoerce (Dict :: Dict ()) :: Dict (filters ~ 1,((outRows - 1) * strideRows) ~ (rows - kernelRows), ((outCols - 1) * strideCols) ~ (cols - kernelCols))) of
          Dict -> do
            (layer  :: Convolution channels 1 kernelRows kernelCols strideRows strideCols) <- createRandomWith wInit gen
            return $ SpecLayer layer (sing :: Sing ('D3 rows cols channels)) (sing :: Sing ('D2 outRows outCols))
      (SNat, SNat, SNat, SNat) ->
        case (unsafeCoerce (Dict :: Dict ()) :: Dict (((outRows - 1) * strideRows) ~ (rows - kernelRows), ((outCols - 1) * strideCols) ~ (cols - kernelCols))) of
          Dict -> do
            (layer  :: Convolution channels filters kernelRows kernelCols strideRows strideCols) <- createRandomWith wInit gen
            return $ SpecLayer layer (sing :: Sing ('D3 rows cols channels)) (sing :: Sing ('D3 outRows outCols filters))


-- | Creates a specification for a convolutional layer with 2D input to the layer. If channels and filters are both 1 then the output is 2D otherwise it is 3D. The output sizes are `out = (in -
-- kernel) / stride + 1`, for rows and cols and the depth is filters for 3D output.
specConvolution2DInput ::
     (Integer, Integer) -- ^ Number of input rows.
  -> Integer -- ^ Number of channels, for the first layer this could be RGB for instance.
  -> Integer -- ^ Number of filters, this is the number of channels output by the layer.
  -> Integer -- ^ The number of rows in the kernel filter
  -> Integer -- ^ The number of column in the kernel filter
  -> Integer -- ^ The row stride of the convolution filter
  -> Integer -- ^ The cols stride of the convolution filter
  -> SpecNet
specConvolution2DInput (rows, cols) = specConvolution3DInput (rows, cols, 1)

-- | Creates a specification for a convolutional layer with 3D input to the layer. If the filter is 1 then the output is 2D, otherwise it is 3D. The output sizes are `out = (in - kernel) / stride +
-- 1`, for rows and cols and the depth is filters for 3D output.
specConvolution3DInput ::
     (Integer, Integer, Integer) -- ^ Input to layer (rows, cols, depths). Use 1 if not used or the function @specConvolution1DInput@ and @specConvolution2DInput@.
  -> Integer -- ^ Number of channels, for the first layer this could be RGB for instance.
  -> Integer -- ^ Number of filters, this is the number of channels output by the layer.
  -> Integer -- ^ The number of rows in the kernel filter
  -> Integer -- ^ The number of column in the kernel filter
  -> Integer -- ^ The row stride of the convolution filter
  -> Integer -- ^ The cols stride of the convolution filter
  -> SpecNet
specConvolution3DInput inp channels filters kernelRows kernelCols strideRows strideCols =
  SpecNetLayer $ SpecConvolution inp channels filters kernelRows kernelCols strideRows strideCols

-- | A convolution layer. 2D and 3D input/output only!
convolution ::
     Integer -- ^ Number of channels, for the first layer this could be RGB for instance.
  -> Integer -- ^ Number of filters, this is the number of channels output by the layer.
  -> Integer -- ^ The number of rows in the kernel filter
  -> Integer -- ^ The number of column in the kernel filter
  -> Integer -- ^ The row stride of the convolution filter
  -> Integer -- ^ The cols stride of the convolution filter
  -> BuildM ()
convolution channels filters kernelRows kernelCols strideRows strideCols = do
  inp@(r, c, _) <- buildRequireLastLayerOut IsNot1D
  let outRows = (r - kernelRows) `div` strideRows + 1
      outCols = (c - kernelCols) `div` strideCols + 1
  buildAddSpec $ SpecNetLayer $ SpecConvolution inp channels filters kernelRows kernelCols strideRows strideCols
  buildSetLastLayer (outRows, outCols, filters)

-------------------- GNum instances --------------------

instance (KnownNat striCols, KnownNat strideRows, KnownNat kernelCols, KnownNat kernelRows, KnownNat filters, KnownNat channels, KnownNat ((kernelRows * kernelCols) * channels)) =>
         GNum (Convolution channels filters kernelRows kernelCols strideRows striCols) where
  n |* (Convolution w store) = Convolution (dmmap (fromRational n *) w) (n |* store)
  (Convolution w1 store1) |+ (Convolution w2 store2) = Convolution (w1 + w2) (store1 |+ store2)
  gFromRational r = Convolution (fromRational r) mkListStore


instance (KnownNat striCols, KnownNat strideRows, KnownNat kernelCols, KnownNat kernelRows, KnownNat filters, KnownNat channels, KnownNat ((kernelRows * kernelCols) * channels)) =>
         GNum (Convolution' channels filters kernelRows kernelCols strideRows striCols) where
  n |* (Convolution' g) = Convolution' (dmmap (fromRational n *) g)
  (Convolution' g) |+ (Convolution' g2) = Convolution' (g + g2)
  gFromRational r = Convolution' (fromRational r)

