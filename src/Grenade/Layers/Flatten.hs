{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE StandaloneDeriving    #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances     #-}

module Grenade.Layers.Flatten (
    FlattenLayer (..)
  ) where

import           Data.Proxy
import           Data.Singletons.TypeLits
import           GHC.TypeLits

import           Numeric.LinearAlgebra.Static
import           Numeric.LinearAlgebra.Data as  LA (flatten, toList, takesV, reshape, vjoin)

import           Grenade.Core.Vector
import           Grenade.Core.Shape
import           Grenade.Core.Network

data FlattenLayer = FlattenLayer
  deriving Show

instance Monad m => UpdateLayer m FlattenLayer where
  type Gradient FlattenLayer = ()
  runUpdate _ _ _ = return FlattenLayer

instance (Monad m, KnownNat a, KnownNat x, KnownNat y, a ~ (x * y)) => Layer m FlattenLayer ('D2 x y) ('D1 a) where
  runForwards _ (S2D' y)   = return $ S1D' . fromList . toList . flatten . extract $ y
  runBackards _ _ (S1D' y) = return ((), S2D' . fromList . toList . unwrap $ y)

instance (Monad m, KnownNat a, KnownNat x, KnownNat y, KnownNat z, a ~ (x * y * z)) => Layer m FlattenLayer ('D3 x y z) ('D1 a) where
  runForwards _ (S3D' y)     = return $ S1D' . raiseShapeError . create . vjoin . vecToList . fmap (flatten . extract) $ y
  runBackards _ _ (S1D' o) = do
    let x'     = fromIntegral $ natVal (Proxy :: Proxy x)
        y'     = fromIntegral $ natVal (Proxy :: Proxy y)
        z'     = fromIntegral $ natVal (Proxy :: Proxy z)
        vecs   = takesV (replicate z' (x' * y')) (extract o)
        ls     = fmap (raiseShapeError . create . reshape y') vecs
        ls'    = mkVector ls :: Vector z (L x y)
    return ((), S3D' ls')

raiseShapeError :: Maybe a -> a
raiseShapeError (Just x) = x
raiseShapeError Nothing = error "Static shape creation from Flatten layer produced the wrong result"
