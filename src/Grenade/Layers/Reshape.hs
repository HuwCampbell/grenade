{-# LANGUAGE CPP                   #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE DeriveAnyClass        #-}
{-# LANGUAGE DeriveGeneric         #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE UndecidableInstances  #-}
{-|
Module      : Grenade.Layers.Reshape
Description : Multipurpose reshaping layer
Copyright   : (c) Huw Campbell, 2016-2017
License     : BSD2
Stability   : experimental
-}
module Grenade.Layers.Reshape
  ( Reshape(..)
  , SpecReshape (..)
  , specReshape
  , specReshape3D1D
  , specReshape3D2D
  , specReshape2D1D
  , specReshape2D3D
  , specReshape1D2D
  , specReshape1D3D
  ) where

#if MIN_VERSION_singletons(2,6,0)
import           Data.Singletons.TypeLits     (SNat (..))
#endif


import           Data.Serialize

import           Control.DeepSeq              (NFData (..))
import           Data.Constraint              (Dict (..))
import           Data.Reflection              (reifyNat)
import           Data.Singletons
import           Data.Singletons.Prelude.Num  ((%*))
import           GHC.Generics                 (Generic)
import           GHC.TypeLits
import           Numeric.LinearAlgebra.Data   as LA (flatten)
import           Numeric.LinearAlgebra.Static
import           Unsafe.Coerce                (unsafeCoerce)


import           Grenade.Core
import           Grenade.Layers.Trivial

-- | Reshape Layer
--
-- The Reshape layer can flatten any 2D or 3D image to 1D vector with the
-- same number of activations, as well as cast up from 1D to a 2D or 3D
-- shape.
--
-- Can also be used to turn a 3D image with only one channel into a 2D image
-- or vice versa.
data Reshape = Reshape
  deriving (Show,Generic,NFData)

instance UpdateLayer Reshape where
  type Gradient Reshape = ()
  runUpdate _ _ _ = Reshape

instance RandomLayer Reshape where
  createRandomWith _ _ = return Reshape

instance (KnownNat a, KnownNat x, KnownNat y, a ~ (x * y)) => Layer Reshape ('D2 x y) ('D1 a) where
  type Tape Reshape ('D2 x y) ('D1 a) = ()
  runForwards _ (S2D y)   =  ((), fromJust' . fromStorable . flatten . extract $ y)
  runBackwards _ _ (S1D y) = ((), fromJust' . fromStorable . extract $ y)

instance (KnownNat a, KnownNat x, KnownNat y, KnownNat (x * z), KnownNat z, a ~ (x * y * z)) => Layer Reshape ('D3 x y z) ('D1 a) where
  type Tape Reshape ('D3 x y z) ('D1 a) = ()
  runForwards _ (S3D y)     = ((), fromJust' . fromStorable . flatten . extract $ y)
  runBackwards _ _ (S1D y) = ((), fromJust' . fromStorable . extract $ y)

instance (KnownNat y, KnownNat x, KnownNat z, z ~ 1) => Layer Reshape ('D3 x y z) ('D2 x y) where
  type Tape Reshape ('D3 x y z) ('D2 x y) = ()
  runForwards _ (S3D y)    = ((), S2D y)
  runBackwards _ _ (S2D y) = ((), S3D y)

instance (KnownNat y, KnownNat x, KnownNat z, z ~ 1) => Layer Reshape ('D2 x y) ('D3 x y z) where
  type Tape Reshape ('D2 x y) ('D3 x y z) = ()
  runForwards _ (S2D y)    = ((), S3D y)
  runBackwards _ _ (S3D y) = ((), S2D y)

instance (KnownNat a, KnownNat x, KnownNat y, a ~ (x * y)) => Layer Reshape ('D1 a) ('D2 x y) where
  type Tape Reshape ('D1 a) ('D2 x y) = ()
  runForwards _ (S1D y)   =  ((), fromJust' . fromStorable . extract $ y)
  runBackwards _ _ (S2D y) = ((), fromJust' . fromStorable . flatten . extract $ y)

instance (KnownNat a, KnownNat x, KnownNat y, KnownNat (x * z), KnownNat z, a ~ (x * y * z)) => Layer Reshape ('D1 a) ('D3 x y z) where
  type Tape Reshape ('D1 a) ('D3 x y z) = ()
  runForwards _ (S1D y)     = ((), fromJust' . fromStorable . extract $ y)
  runBackwards _ _ (S3D y)  = ((), fromJust' . fromStorable . flatten . extract $ y)

instance Serialize Reshape where
  put _ = return ()
  get = return Reshape


fromJust' :: Maybe x -> x
fromJust' (Just x) = x
fromJust' Nothing  = error "Reshape error: data shape couldn't be converted."

-------------------- DynamicNetwork instance --------------------

instance FromDynamicLayer Reshape where
  fromDynamicLayer inShape outShape _ =
     SpecNetLayer $ SpecReshape (tripleFromSomeShape inShape) (tripleFromSomeShape outShape)

instance ToDynamicLayer SpecReshape where
  toDynamicLayer _ _ (SpecReshape inp@(rowsI, colsI, depthI) out@(rowsO, colsO, depthO)) =
    reifyNat rowsI $ \(pxRI :: (KnownNat rI) => Proxy rI) ->
    reifyNat colsI $ \(_ :: (KnownNat cI) => Proxy cI) ->
    reifyNat depthI $ \(pxDI :: (KnownNat dI) => Proxy dI) ->
    reifyNat rowsO $ \(pxRO:: (KnownNat rO) => Proxy rO) ->
    reifyNat colsO $ \(_ :: (KnownNat cO) => Proxy cO) ->
    reifyNat depthO $ \(pxDO :: (KnownNat dO) => Proxy dO) ->
    case (singByProxy pxRI %* singByProxy pxDI, singByProxy pxRO %* singByProxy pxDO) of
          (SNat, SNat) ->
            if (rowsI, colsI, depthI) == (rowsO, colsO, depthO)
            then case (rowsI, colsI, depthI) of
              (_,1,1) -> return $ SpecLayer Trivial (sing :: Sing ('D1 rI)) (sing :: Sing ('D1 rI))
              (_,_,1) -> return $ SpecLayer Trivial (sing :: Sing ('D2 rI cI)) (sing :: Sing ('D2 rI cI))
              (_,_,_) -> return $ SpecLayer Trivial (sing :: Sing ('D3 rI cI dI)) (sing :: Sing ('D3 rI cI dI))
            else
            case ((rowsI, colsI, depthI), (rowsO, colsO, depthO)) of
              ((_,1,1), (_, _, 1)) | rowsI /= rowsO * colsO -> err
              ((_,1,1), (_, _, 1)) ->
                case (unsafeCoerce (Dict::Dict()) :: Dict (rI ~ (rO * cO))) of
                  Dict -> return $ SpecLayer Reshape (sing :: Sing ('D1 rI)) (sing :: Sing ('D2 rO cO))
              ((_,1,1), (_, _, _)) | rowsI /= rowsO * colsO * depthO -> err
              ((_,1,1), (_, _, _)) ->
                case (unsafeCoerce (Dict::Dict()) :: Dict (rI ~ ((rO * cO) * dO))) of
                  Dict -> return $ SpecLayer Reshape (sing :: Sing ('D1 rI)) (sing :: Sing ('D3 rO cO dO))
              ((_,_,1), (_, 1, 1)) | rowsI * colsI /= rowsO -> err
              ((_,_,1), (_, 1, 1)) ->
                case (unsafeCoerce (Dict::Dict()) :: Dict ((rI * cI) ~ rO)) of
                  Dict -> return $ SpecLayer Reshape (sing :: Sing ('D2 rI cI)) (sing :: Sing ('D1 rO))
              ((_,_,1), (_, _, _)) | 1 /= depthO -> err
              ((_,_,1), (_, _, _)) ->
                case (unsafeCoerce (Dict::Dict()) :: Dict (dO ~ 1)) of
                  Dict -> return $ SpecLayer Reshape (sing :: Sing ('D2 rI cI)) (sing :: Sing ('D3 rI cI dO))
              ((_,_,_), (_, 1, 1)) | rowsI * colsI * depthO /= rowsO -> err
              ((_,_,_), (_, 1, 1)) ->
                case (unsafeCoerce (Dict::Dict()) :: Dict (((rI * cI) * dI) ~ rO)) of
                  Dict -> return $ SpecLayer Reshape (sing :: Sing ('D3 rI cI dI)) (sing :: Sing ('D1 rO))
              ((_,_,_), (_, _, 1)) | depthI == 1 -> err
              ((_,_,_), (_, _, 1)) ->
                case (unsafeCoerce (Dict::Dict()) :: Dict (dI ~ 1)) of
                  Dict -> return $ SpecLayer Reshape (sing :: Sing ('D3 rI cI dI)) (sing :: Sing ('D2 rI cI))
              (_, _) -> error $ "Reshaping using a specificaiton from " ++ show inp ++ " to " ++ show out ++ " is not possible!"
    where err = error $ "cannot reshape from " ++ show inp ++ " to " ++ show out ++ ". Sizes (number of elements) do not match or it is trivial."


specReshape :: (Integer, Integer, Integer) -> (Integer, Integer, Integer) -> SpecNet
specReshape inp out = SpecNetLayer $ SpecReshape inp out

specReshape3D2D :: (Integer, Integer, Integer) -> (Integer, Integer) -> SpecNet
specReshape3D2D inp (rows, cols) = specReshape inp (rows, cols, 1)

specReshape3D1D :: (Integer, Integer, Integer) -> Integer -> SpecNet
specReshape3D1D inp rows = specReshape inp (rows, 1, 1)

specReshape2D3D :: (Integer, Integer) -> (Integer, Integer, Integer) -> SpecNet
specReshape2D3D (rows, cols) = specReshape (rows, cols, 1)


specReshape2D1D :: (Integer, Integer) -> Integer -> SpecNet
specReshape2D1D (rowsI, colsI) rowsO = specReshape (rowsI, colsI, 1) (rowsO, 1, 1)

specReshape1D3D :: Integer -> (Integer, Integer, Integer) -> SpecNet
specReshape1D3D rows = specReshape (rows, 1, 1)

specReshape1D2D :: Integer -> (Integer, Integer) -> SpecNet
specReshape1D2D rowsI (rowsO, colsO) = specReshape (rowsI, 1, 1) (rowsO, colsO, 1)


-------------------- GNum instances --------------------


instance GNum Reshape where
  _ |* Reshape = Reshape
  _ |+ Reshape = Reshape
  gFromRational _ = Reshape
