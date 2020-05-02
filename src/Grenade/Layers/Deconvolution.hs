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
Module      : Grenade.Layers.Deconvolution
Description : Deconvolution layer
Copyright   : (c) Huw Campbell, 2016-2017
License     : BSD2
Stability   : experimental

A deconvolution layer is in many ways a convolution layer in reverse.
It learns a kernel to apply to each pixel location, spreading it out
into a larger layer.

This layer is important for image generation tasks, such as GANs on
images.
-}
module Grenade.Layers.Deconvolution (
    Deconvolution (..)
  , Deconvolution' (..)
  , SpecDeconvolution (..)
  , specDeconvolution2DInput
  , specDeconvolution3DInput
  , deconvolution
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
import           Unsafe.Coerce

import           Grenade.Core
import           Grenade.Dynamic
import           Grenade.Dynamic.Internal.Build
import           Grenade.Layers.Internal.Convolution
import           Grenade.Layers.Internal.Update
import           Grenade.Utils.LinearAlgebra
import           Grenade.Utils.ListStore

-- | A Deconvolution layer for a neural network.
--   This uses the im2col Convolution trick popularised by Caffe.
--
--   The Deconvolution layer is a way of spreading out a single response
--   into a larger image, and is useful in generating images.
--
data Deconvolution :: Nat -- Number of channels, for the first layer this could be RGB for instance.
                   -> Nat -- Number of filters, this is the number of channels output by the layer.
                   -> Nat -- The number of rows in the kernel filter
                   -> Nat -- The number of column in the kernel filter
                   -> Nat -- The row stride of the Deconvolution filter
                   -> Nat -- The columns stride of the Deconvolution filter
                   -> Type where
  Deconvolution :: ( KnownNat channels
                   , KnownNat filters
                   , KnownNat kernelRows
                   , KnownNat kernelColumns
                   , KnownNat strideRows
                   , KnownNat strideColumns
                   , KnownNat kernelFlattened
                   , kernelFlattened ~ (kernelRows * kernelColumns * filters))
                 => !(L kernelFlattened channels) -- The kernel filter weights
                 -> !(ListStore (L kernelFlattened channels)) -- The last kernel update (or momentum)
                 -> Deconvolution channels filters kernelRows kernelColumns strideRows strideColumns

instance NFData (Deconvolution channels filters kernelRows kernelColumns strideRows strideColumns) where
  rnf (Deconvolution a b) = rnf a `seq` rnf b

instance ( KnownNat channels
         , KnownNat filters
         , KnownNat kernelRows
         , KnownNat kernelColumns
         , KnownNat strideRows
         , KnownNat strideColumns
         , KnownNat (kernelRows * kernelColumns * filters)
         ) => Serialize (Deconvolution channels filters kernelRows kernelColumns strideRows strideColumns) where
  put (Deconvolution w store) = do
    putListOf put . toList . flatten . extract $ w
    put (fmap (toList . flatten . extract) store)
  get = do
      let f  = fromIntegral $ natVal (Proxy :: Proxy channels)
      wN    <- maybe (fail "Vector of incorrect size") return . create . reshape f . LA.fromList =<< getListOf get
      store <- fmap (fromMaybe (error "Vector of incorrect size") . create . reshape f . LA.fromList)  <$> get
      return $ Deconvolution wN store


data Deconvolution' :: Nat -- Number of channels, for the first layer this could be RGB for instance.
                    -> Nat -- Number of filters, this is the number of channels output by the layer.
                    -> Nat -- The number of rows in the kernel filter
                    -> Nat -- The number of column in the kernel filter
                    -> Nat -- The row stride of the Deconvolution filter
                    -> Nat -- The columns stride of the Deconvolution filter
                    -> Type where
  Deconvolution' :: ( KnownNat channels
                    , KnownNat filters
                    , KnownNat kernelRows
                    , KnownNat kernelColumns
                    , KnownNat strideRows
                    , KnownNat strideColumns
                    , KnownNat kernelFlattened
                    , kernelFlattened ~ (kernelRows * kernelColumns * filters))
                 => !(L kernelFlattened channels) -- The kernel filter gradient
                 -> Deconvolution' channels filters kernelRows kernelColumns strideRows strideColumns

instance NFData (Deconvolution' channels filters kernelRows kernelColumns strideRows strideColumns) where
  rnf (Deconvolution' a) = rnf a `seq` ()

instance ( KnownNat channels
         , KnownNat filters
         , KnownNat kernelRows
         , KnownNat kernelColumns
         , KnownNat strideRows
         , KnownNat strideColumns
         , KnownNat (kernelRows * kernelColumns * filters)
         ) =>
         Serialize (Deconvolution' channels filters kernelRows kernelColumns strideRows strideColumns) where
  put (Deconvolution' w) = putListOf put . toList . flatten . extract $ w
  get = do
    let f = fromIntegral $ natVal (Proxy :: Proxy channels)
    wN <- maybe (fail "Vector of incorrect size") return . create . reshape f . LA.fromList =<< getListOf get
    return $ Deconvolution' wN

instance (KnownNat channels
         , KnownNat filters
         , KnownNat kernelRows
         , KnownNat kernelColumns
         , KnownNat strideRows
         , KnownNat strideColumns) => FoldableGradient (Deconvolution' channels filters kernelRows kernelColumns strideRows strideColumns) where
  mapGradient f (Deconvolution' kernelGradient) = Deconvolution' (dmmap f kernelGradient)
  squaredSums (Deconvolution' kernelGradient) = [sumM . squareM $ kernelGradient]


instance Show (Deconvolution c f k k' s s') where
  show (Deconvolution a _) = renderConv a
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
         , KnownNat ((kernelRows * kernelColumns) * filters)
         , KnownNat ((kernelRows * kernelColumns) * channels)
         , KnownNat (channels * ((kernelRows * kernelColumns) * filters))
         ) =>
         RandomLayer (Deconvolution channels filters kernelRows kernelColumns strideRows strideColumns) where
  createRandomWith m gen = do
    wN <- getRandomMatrix i i m gen
    return $ Deconvolution wN mkListStore
    where
      i = natVal (Proxy :: Proxy ((kernelRows * kernelColumns) * channels))


instance ( KnownNat channels
         , KnownNat filters
         , KnownNat kernelRows
         , KnownNat kernelColumns
         , KnownNat strideRows
         , KnownNat strideColumns
         , KnownNat (kernelRows * kernelColumns * filters)
         ) => UpdateLayer (Deconvolution channels filters kernelRows kernelColumns strideRows strideColumns) where
  type Gradient (Deconvolution channels filters kernelRows kernelColumns strideRows strideColumns) = (Deconvolution' channels filters kernelRows kernelColumns strideRows strideColumns)

  type MomentumStore (Deconvolution channels filters kernelRows kernelColumns strideRows strideColumns)  = ListStore (L (kernelRows * kernelColumns * filters) channels)

  runUpdate opt@OptSGD{} x@(Deconvolution oldKernel store) (Deconvolution' kernelGradient) =
    let oldMomentum = getData opt x store
        result = descendMatrix opt (MatrixValuesSGD oldKernel kernelGradient oldMomentum)
        newStore = setData opt x store (matrixMomentum result)
    in Deconvolution (matrixActivations result) newStore
  runUpdate opt@OptAdam{} x@(Deconvolution oldKernel store) (Deconvolution' kernelGradient) =
    let (m, v) = toTuple $ getData opt x store
        result = descendMatrix opt (MatrixValuesAdam (getStep store) oldKernel kernelGradient m v)
        newStore = setData opt x store [matrixM result, matrixV result]
    in Deconvolution (matrixActivations result) newStore
    where toTuple [m ,v] = (m, v)
          toTuple xs = error $ "unexpected input of length " ++ show (length xs) ++ "in toTuple in Convolution.hs"

instance ( KnownNat channels
         , KnownNat filters
         , KnownNat kernelRows
         , KnownNat kernelColumns
         , KnownNat strideRows
         , KnownNat strideColumns
         , KnownNat (kernelRows * kernelColumns * filters)
         ) =>
         LayerOptimizerData (Deconvolution channels filters kernelRows kernelColumns strideRows strideColumns) (Optimizer 'SGD) where
  type MomentumExpOptResult (Deconvolution channels filters kernelRows kernelColumns strideRows strideColumns) (Optimizer 'SGD) = L (kernelRows * kernelColumns * filters) channels
  type MomentumDataType (Deconvolution channels filters kernelRows kernelColumns strideRows strideColumns) (Optimizer 'SGD) = L (kernelRows * kernelColumns * filters) channels
  getData opt x store = head $ getListStore opt x store
  setData opt x store = setListStore opt x store . return
  newData _ _ = konst 0

instance ( KnownNat channels
         , KnownNat filters
         , KnownNat kernelRows
         , KnownNat kernelColumns
         , KnownNat strideRows
         , KnownNat strideColumns
         , KnownNat (kernelRows * kernelColumns * filters)
         ) => LayerOptimizerData (Deconvolution channels filters kernelRows kernelColumns strideRows strideColumns) (Optimizer 'Adam) where
  type MomentumExpOptResult (Deconvolution channels filters kernelRows kernelColumns strideRows strideColumns) (Optimizer 'Adam)  = [L (kernelRows * kernelColumns * filters) channels]
  type MomentumDataType (Deconvolution channels filters kernelRows kernelColumns strideRows strideColumns) (Optimizer 'Adam) = L (kernelRows * kernelColumns * filters) channels
  getData = getListStore
  setData = setListStore
  newData _ _ = konst 0


-- | A two dimentional image may have a Deconvolution filter applied to it
instance ( KnownNat kernelRows
         , KnownNat kernelCols
         , KnownNat filters
         , KnownNat strideRows
         , KnownNat strideCols
         , KnownNat inputRows
         , KnownNat inputCols
         , KnownNat outputRows
         , KnownNat outputCols
         , ((inputRows - 1) * strideRows) ~ (outputRows - kernelRows)
         , ((inputCols - 1) * strideCols) ~ (outputCols - kernelCols)
         , KnownNat (kernelRows * kernelCols * filters)
         , KnownNat (outputRows * filters)
         ) => Layer (Deconvolution 1 filters kernelRows kernelCols strideRows strideCols) ('D2 inputRows inputCols) ('D3 outputRows outputCols filters) where
  type Tape (Deconvolution 1 filters kernelRows kernelCols strideRows strideCols) ('D2 inputRows inputCols) ('D3 outputRows outputCols filters) = S ('D3 inputRows inputCols 1)
  runForwards c (S2D input) =
    runForwards c (S3D input :: S ('D3 inputRows inputCols 1))

  runBackwards c tape grads =
    case runBackwards c tape grads of
      (c', S3D back :: S ('D3 inputRows inputCols 1)) ->  (c', S2D back)

-- | A two dimentional image may have a Deconvolution filter applied to it
instance ( KnownNat kernelRows
         , KnownNat kernelCols
         , KnownNat strideRows
         , KnownNat strideCols
         , KnownNat inputRows
         , KnownNat inputCols
         , KnownNat outputCols
         , ((inputRows - 1) * strideRows) ~ (outputRows - kernelRows)
         , ((inputCols - 1) * strideCols) ~ (outputCols - kernelCols)
         , KnownNat (kernelRows * kernelCols * 1)
         , KnownNat (outputRows * 1)
         ) => Layer (Deconvolution 1 1 kernelRows kernelCols strideRows strideCols) ('D2 inputRows inputCols) ('D2 outputRows outputCols) where
  type Tape (Deconvolution 1 1 kernelRows kernelCols strideRows strideCols) ('D2 inputRows inputCols) ('D2 outputRows outputCols) = S ('D3 inputRows inputCols 1)
  runForwards c (S2D input) =
    case runForwards c (S3D input :: S ('D3 inputRows inputCols 1)) of
      (tps, S3D fore :: S ('D3 outputRows outputCols 1)) ->  (tps, S2D fore)

  runBackwards c tape (S2D grads) =
    case runBackwards c tape (S3D grads :: S ('D3 outputRows outputCols 1)) of
      (c', S3D back :: S ('D3 inputRows inputCols 1)) ->  (c', S2D back)

-- | A two dimentional image may have a Deconvolution filter applied to it
instance ( KnownNat kernelRows
         , KnownNat kernelCols
         , KnownNat strideRows
         , KnownNat strideCols
         , KnownNat inputRows
         , KnownNat inputCols
         , KnownNat outputCols
         , ((inputRows - 1) * strideRows) ~ (outputRows - kernelRows)
         , ((inputCols - 1) * strideCols) ~ (outputCols - kernelCols)
         , KnownNat (kernelRows * kernelCols * 1)
         , KnownNat (outputRows * 1)
         , KnownNat channels
         ) => Layer (Deconvolution channels 1 kernelRows kernelCols strideRows strideCols) ('D3 inputRows inputCols channels) ('D2 outputRows outputCols) where
  type Tape (Deconvolution channels 1 kernelRows kernelCols strideRows strideCols) ('D3 inputRows inputCols channels) ('D2 outputRows outputCols) = S ('D3 inputRows inputCols channels)
  runForwards c input =
    case runForwards c input of
      (tps, S3D fore :: S ('D3 outputRows outputCols 1)) ->  (tps, S2D fore)

  runBackwards c tape (S2D grads) =
    runBackwards c tape (S3D grads :: S ('D3 outputRows outputCols 1))

-- | A three dimensional image (or 2d with many channels) can have
--   an appropriately sized Deconvolution filter run across it.
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
         , ((inputRows - 1) * strideRows) ~ (outputRows - kernelRows)
         , ((inputCols - 1) * strideCols) ~ (outputCols - kernelCols)
         , KnownNat (kernelRows * kernelCols * filters)
         , KnownNat (outputRows * filters)
         ) => Layer (Deconvolution channels filters kernelRows kernelCols strideRows strideCols) ('D3 inputRows inputCols channels) ('D3 outputRows outputCols filters) where

  type Tape (Deconvolution channels filters kernelRows kernelCols strideRows strideCols) ('D3 inputRows inputCols channels) ('D3 outputRows outputCols filters) = S ('D3 inputRows inputCols channels)

  runForwards (Deconvolution kernel _) (S3D input) =
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

        c  = vid2col 1 1 1 1 ix iy ex

        mt = c LA.<> tr ek

        r  = col2vid kx ky sx sy ox oy mt
        rs = fromJust . create $ r
    in  (S3D input, S3D rs)
  runBackwards (Deconvolution kernel _) (S3D input) (S3D dEdy) =
    let ex = extract input
        ix = fromIntegral $ natVal (Proxy :: Proxy inputRows)
        iy = fromIntegral $ natVal (Proxy :: Proxy inputCols)
        kx = fromIntegral $ natVal (Proxy :: Proxy kernelRows)
        ky = fromIntegral $ natVal (Proxy :: Proxy kernelCols)
        sx = fromIntegral $ natVal (Proxy :: Proxy strideRows)
        sy = fromIntegral $ natVal (Proxy :: Proxy strideCols)
        ox = fromIntegral $ natVal (Proxy :: Proxy outputRows)
        oy = fromIntegral $ natVal (Proxy :: Proxy outputCols)

        c  = vid2col 1 1 1 1 ix iy ex

        eo = extract dEdy
        ek = extract kernel

        vs = vid2col kx ky sx sy ox oy eo

        kN = fromJust . create . tr $ tr c LA.<> vs

        dW = vs LA.<> ek

        xW = col2vid 1 1 1 1 ix iy dW
    in  (Deconvolution' kN, S3D . fromJust . create $ xW)


-------------------- DynamicNetwork instance --------------------

instance (KnownNat channels, KnownNat filters, KnownNat kernelRows, KnownNat kernelColumns, KnownNat strideRows, KnownNat strideColumns) =>
         FromDynamicLayer (Deconvolution channels filters kernelRows kernelColumns strideRows strideColumns) where
  fromDynamicLayer inp _ _ =
    SpecNetLayer $
    SpecDeconvolution
      (tripleFromSomeShape inp)
      (natVal (Proxy :: Proxy channels))
      (natVal (Proxy :: Proxy filters))
      (natVal (Proxy :: Proxy kernelRows))
      (natVal (Proxy :: Proxy kernelColumns))
      (natVal (Proxy :: Proxy strideRows))
      (natVal (Proxy :: Proxy strideColumns))


instance ToDynamicLayer SpecDeconvolution where
  toDynamicLayer  = toDynamicLayer'

toDynamicLayer' :: (PrimBase m) => WeightInitMethod -> Gen (PrimState m) -> SpecDeconvolution -> m SpecNetwork
toDynamicLayer' _ _ (SpecDeconvolution inp@(_, 1, 1) _ _ _ _ _ _) = error $ "1D input to a deconvolutional layer is not permited! you specified: " ++ show inp
toDynamicLayer' wInit gen (SpecDeconvolution (rows, cols, depth) ch fil kerRows kerCols strRows strCols) =
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
    case ( (singByProxy pxKerRows %* singByProxy pxKerCols) %* singByProxy pxFil
         , (singByProxy pxKerRows %* singByProxy pxKerCols) %* singByProxy pxCh -- this is the input: i = (kernelRows * kernelCols) * channels)
         , singByProxy pxCh %* ((singByProxy pxKerRows %* singByProxy pxKerCols) %* singByProxy pxFil)
         , singByProxy pxOutRows %* singByProxy pxFil -- 'D3 representation
         , singByProxy pxRows %* singByProxy pxCh -- 'D3 representation
         ) of
      (SNat, SNat, SNat, SNat, SNat) | ch == 1 && fil == 1 && depth == 0 ->
        case (unsafeCoerce (Dict :: Dict()) :: Dict (channels ~ 1, filters ~ 1, ((rows - 1) * strideRows) ~ (outRows - kernelRows), ((cols - 1) * strideCols) ~ (outCols - kernelCols)) ) of
          Dict -> do
            (layer  :: Deconvolution 1 1 kernelRows kernelCols strideRows strideCols) <- createRandomWith wInit gen
            return $ SpecLayer layer (sing :: Sing ('D2 rows cols)) (sing :: Sing ('D2 outRows outCols))
      (SNat, SNat, SNat, SNat, SNat) | ch == 1 ->
        case (unsafeCoerce (Dict :: Dict()) :: Dict (channels ~ 1, ((rows - 1) * strideRows) ~ (outRows - kernelRows), ((cols - 1) * strideCols) ~ (outCols - kernelCols)) ) of
          Dict -> do
            (layer  :: Deconvolution 1 filters kernelRows kernelCols strideRows strideCols) <- createRandomWith wInit gen
            return $ SpecLayer layer (sing :: Sing ('D2 rows cols)) (sing :: Sing ('D3 outRows outCols filters))
      (SNat, SNat, SNat, SNat, SNat) | fil == 1 ->
        case (unsafeCoerce (Dict :: Dict()) :: Dict (filters ~ 1, ((rows - 1) * strideRows) ~ (outRows - kernelRows), ((cols - 1) * strideCols) ~ (outCols - kernelCols)) ) of
          Dict -> do
            (layer  :: Deconvolution channels 1 kernelRows kernelCols strideRows strideCols) <- createRandomWith wInit gen
            return $ SpecLayer layer (sing :: Sing ('D3 rows cols channels)) ( sing :: Sing ('D2 outRows outCols))
      (SNat, SNat, SNat, SNat, SNat) ->
        case (unsafeCoerce (Dict :: Dict()) :: Dict (((rows - 1) * strideRows) ~ (outRows - kernelRows), ((cols - 1) * strideCols) ~ (outCols - kernelCols)) ) of
          Dict -> do
            (layer :: Deconvolution channels filters kernelRows kernelCols strideRows strideCols) <- createRandomWith wInit gen
            return $ SpecLayer layer (sing :: Sing ('D3 rows cols channels)) ( sing :: Sing ('D3 outRows outCols filters))


-- | Creates a specification for a deconvolutional layer with 2D input to the layer. If channels and filters are both 1 then the output is 2D otherwise it is 3D. The output sizes are `out = (in - 1) *
-- stride + kernel`, for rows and cols and the depth is filters for 3D output.
specDeconvolution2DInput ::
     (Integer, Integer) -- ^ Number of input rows.
  -> Integer -- ^ Number of channels, for the first layer this could be RGB for instance.
  -> Integer -- ^ Number of filters, this is the number of channels output by the layer.
  -> Integer -- ^ The number of rows in the kernel filter
  -> Integer -- ^ The number of column in the kernel filter
  -> Integer -- ^ The row stride of the deconvolution filter
  -> Integer -- ^ The cols stride of the deconvolution filter
  -> SpecNet
specDeconvolution2DInput (rows, cols) = specDeconvolution3DInput (rows, cols, 1)

-- | Creates a specification for a deconvolutional layer with 3D input to the layer. If the filter is 1 then the output is 2D, otherwise it is 3D. The output sizes are `out = (in - 1) * stride +
-- kernel`, for rows and cols and the depth is filters for 3D output.
specDeconvolution3DInput ::
     (Integer, Integer, Integer) -- ^ Input to layer (rows, cols, depths). Use 1 if not used or the function @specDeconvolution1DInput@ and @specDeconvolution2DInput@.
  -> Integer -- ^ Number of channels, for the first layer this could be RGB for instance.
  -> Integer -- ^ Number of filters, this is the number of channels output by the layer.
  -> Integer -- ^ The number of rows in the kernel filter
  -> Integer -- ^ The number of column in the kernel filter
  -> Integer -- ^ The row stride of the deconvolution filter
  -> Integer -- ^ The cols stride of the deconvolution filter
  -> SpecNet
specDeconvolution3DInput inp channels filters kernelRows kernelCols strideRows strideCols =
  SpecNetLayer $ SpecDeconvolution inp channels filters kernelRows kernelCols strideRows strideCols


-- | A deconvolution layer. 2D and 3D input/output only!
deconvolution ::
     Integer -- ^ Number of channels, for the first layer this could be RGB for instance.
  -> Integer -- ^ Number of filters, this is the number of channels output by the layer.
  -> Integer -- ^ The number of rows in the kernel filter
  -> Integer -- ^ The number of column in the kernel filter
  -> Integer -- ^ The row stride of the deconvolution filter
  -> Integer -- ^ The cols stride of the deconvolution filter
  -> BuildM ()
deconvolution channels filters kernelRows kernelCols strideRows strideCols = do
  inp@(r, c, _) <- buildRequireLastLayerOut IsNot1D
  let outRows = (r - 1) * strideRows + kernelRows
      outCols = (c - 1) * strideCols + kernelCols
  buildAddSpec $ SpecNetLayer $ SpecDeconvolution inp channels filters kernelRows kernelCols strideRows strideCols
  buildSetLastLayer (outRows, outCols, filters)

-------------------- GNum instances --------------------


instance (KnownNat strideCols, KnownNat strideRows, KnownNat kernelCols, KnownNat kernelRows, KnownNat filters, KnownNat channels, KnownNat ((kernelRows * kernelCols) * filters)) =>
         GNum (Deconvolution channels filters kernelRows kernelCols strideRows strideCols) where
  n |* (Deconvolution w store) = Deconvolution (dmmap (fromRational n *) w) (n |* store)
  (Deconvolution w1 store1) |+ (Deconvolution w2 store2) = Deconvolution (w1 + w2) (store1 |+ store2)
  gFromRational r = Deconvolution (fromRational r) mkListStore


instance (KnownNat strideCols, KnownNat strideRows, KnownNat kernelCols, KnownNat kernelRows, KnownNat filters, KnownNat channels, KnownNat ((kernelRows * kernelCols) * filters)) =>
         GNum (Deconvolution' channels filters kernelRows kernelCols strideRows strideCols) where
  n |* (Deconvolution' g) = Deconvolution' (dmmap (fromRational n *) g)
  (Deconvolution' g) |+ (Deconvolution' g2) = Deconvolution' (g + g2)
  gFromRational r = Deconvolution' (fromRational r)
