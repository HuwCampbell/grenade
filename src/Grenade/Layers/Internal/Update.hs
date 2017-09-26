{-# LANGUAGE ForeignFunctionInterface #-}
module Grenade.Layers.Internal.Update (
    descendMatrix
  , descendVector
  ) where

import           Data.Maybe ( fromJust )
import qualified Data.Vector.Storable as U ( unsafeToForeignPtr0, unsafeFromForeignPtr0 )

import           Foreign ( mallocForeignPtrArray, withForeignPtr )
import           Foreign.Ptr ( Ptr )
import           GHC.TypeLits

import           Numeric.LinearAlgebra ( Vector, flatten )
import           Numeric.LinearAlgebra.Static
import qualified Numeric.LinearAlgebra.Devel as U

import           System.IO.Unsafe ( unsafePerformIO )

descendMatrix :: (KnownNat rows, KnownNat columns) => Double -> Double -> Double -> L rows columns -> L rows columns -> L rows columns -> (L rows columns, L rows columns)
descendMatrix rate momentum regulariser weights gradient lastUpdate =
  let (rows, cols) = size weights
      len          = rows * cols
      -- Most gradients come in in ColumnMajor,
      -- so we'll transpose here before flattening them
      -- into a vector to prevent a copy.
      --
      -- This gives ~15% speed improvement for LSTMs.
      weights'     = flatten . tr . extract $ weights
      gradient'    = flatten . tr . extract $ gradient
      lastUpdate'  = flatten . tr . extract $ lastUpdate
      (vw, vm)     = descendUnsafe len rate momentum regulariser weights' gradient' lastUpdate'

      -- Note that it's ColumnMajor, as we did a transpose before
      -- using the internal vectors.
      mw           = U.matrixFromVector U.ColumnMajor rows cols vw
      mm           = U.matrixFromVector U.ColumnMajor rows cols vm
  in  (fromJust . create $ mw, fromJust . create $ mm)

descendVector :: (KnownNat r) => Double -> Double -> Double -> R r -> R r -> R r -> (R r, R r)
descendVector rate momentum regulariser weights gradient lastUpdate =
  let len          = size weights
      weights'     = extract weights
      gradient'    = extract gradient
      lastUpdate'  = extract lastUpdate
      (vw, vm)     = descendUnsafe len rate momentum regulariser weights' gradient' lastUpdate'
  in  (fromJust $ create vw, fromJust $ create vm)

descendUnsafe :: Int -> Double -> Double -> Double -> Vector Double -> Vector Double -> Vector Double -> (Vector Double, Vector Double)
descendUnsafe len rate momentum regulariser weights gradient lastUpdate =
  unsafePerformIO $ do
    outWPtr <- mallocForeignPtrArray len
    outMPtr <- mallocForeignPtrArray len
    let (wPtr, _) = U.unsafeToForeignPtr0 weights
    let (gPtr, _) = U.unsafeToForeignPtr0 gradient
    let (lPtr, _) = U.unsafeToForeignPtr0 lastUpdate

    withForeignPtr wPtr $ \wPtr' ->
      withForeignPtr gPtr $ \gPtr' ->
        withForeignPtr lPtr $ \lPtr' ->
          withForeignPtr outWPtr $ \outWPtr' ->
            withForeignPtr outMPtr $ \outMPtr' ->
              descend_cpu len rate momentum regulariser wPtr' gPtr' lPtr' outWPtr' outMPtr'

    return (U.unsafeFromForeignPtr0 outWPtr len, U.unsafeFromForeignPtr0 outMPtr len)

foreign import ccall unsafe
    descend_cpu
      :: Int -> Double -> Double -> Double -> Ptr Double -> Ptr Double -> Ptr Double -> Ptr Double -> Ptr Double -> IO ()

