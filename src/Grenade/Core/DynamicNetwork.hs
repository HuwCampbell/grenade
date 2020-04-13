{-# LANGUAGE CPP                   #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE InstanceSigs          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PolyKinds           #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE UndecidableInstances  #-}
{-|
Module      : Grenade.Layers.DynamicNetwork
Description : Dynamic grenade networks
Copyright   : (c) Manuel Schneckenreither, 2016-2020
License     : BSD2
Stability   : experimental

This module defines types and functions for dynamic generation of networks.
-}

module Grenade.Core.DynamicNetwork
  ( SpecNetwork (..)
  , FromDynamicLayer (..)
  , ToDynamicLayer (..)
  , SpecNet (..)
  , networkFromSpecification
  -- -- Convenience functions for creating dynamic network specifications:
  , (|=>)
  , specNil1D
  , specNil2D
  , specNil3D
  ) where

import           Control.DeepSeq
import           Control.Monad.Primitive           (PrimBase, PrimState)
import           Data.Constraint                   (Dict (..))
import           Data.Reflection (reifyNat)
import Data.Typeable as T (typeOf, Typeable, cast) 
import           Data.Serialize
import           Data.Singletons
import           Data.Singletons.Prelude
import           GHC.TypeLits
import           System.Random.MWC
import           Unsafe.Coerce                     (unsafeCoerce)

#if MIN_VERSION_base(4,9,0)
import           Data.Kind                         (Type)
#endif

import           Grenade.Core.Network
import           Grenade.Core.Layer
import           Grenade.Core.Shape
import           Grenade.Core.WeightInitialization


-- | Specification of a network or layer.
data SpecNetwork :: Type where
  SpecNetwork
    :: (SingI shapes, SingI shapes, SingI (Head shapes), SingI (Last shapes), Show (Network layers shapes), FromDynamicLayer (Network layers shapes))
    => Network layers shapes
    -> SpecNetwork
  SpecLayer :: (FromDynamicLayer x, Show x) => x -> SomeSing Shape -> SomeSing Shape -> SpecNetwork
  
instance Show SpecNetwork where
  show (SpecNetwork net) = show net
  show (SpecLayer x _ _) = show x

instance FromDynamicLayer SpecNetwork where
  fromDynamicLayer (SpecNetwork net) = fromDynamicLayer net
  fromDynamicLayer (SpecLayer x _ _) = fromDynamicLayer x
  
  
--------------------------------------------------

-- | Create a runtime dynamic specification of a network. Dynamic layers (and networks), for storing and restoring specific network structures (e.g. in saving the network structures to a DB and
-- restoring it from there) or simply generating them at runtime. This does not store the weights and biases! They have to be handled separately (see Serialize)!
class FromDynamicLayer x where
  fromDynamicLayer :: x -> SpecNet

-- | Class for generating layers from a specification.
class (Show spec) => ToDynamicLayer spec where
  toDynamicLayer :: (PrimBase m) => WeightInitMethod -> Gen (PrimState m) -> spec -> m SpecNetwork 


-- Specification of a network and its layers

-- | Data structure for holding specifications for networks. Networks can be built dynamically with @toDynamicLayer@. Further see the functions @|=>@, @specNil1D@, @specNil2D@, @specNil3D@, and
-- possibly any other layer implementation of @ToDynamicLayer@ for building specifications.
data SpecNet 
  = SpecNNil1D !Integer                   -- ^ 1D network output
  | SpecNNil2D !Integer !Integer          -- ^ 2D network output
  | SpecNNil3D !Integer !Integer !Integer -- ^ 3D network output
  | SpecNCons !SpecNet !SpecNet           -- ^ x :~> xs, where x can also be a network
  | forall spec . (ToDynamicLayer spec, Typeable spec, Ord spec, Eq spec, Show spec, Read spec, NFData spec) => SpecNetLayer !spec -- ^ Specification of a layer


instance Eq SpecNet where
  SpecNNil1D r1 == SpecNNil1D r2 = r1 == r2
  SpecNNil2D r1 c1 == SpecNNil2D r2 c2 = r1 == r2 && c1 == c2
  SpecNNil3D r1 c1 d1 == SpecNNil3D r2 c2 d2 = r1 == r2 && c1 == c2 && d1 == d2
  SpecNCons l1 r1 == SpecNCons l2 r2 = l1 == l2 && r1 == r2
  SpecNetLayer spec1 == SpecNetLayer spec2 = T.typeOf spec1 == T.typeOf spec2 && Just spec1 == cast spec2
  _ == _ = False

instance Ord SpecNet where
  SpecNNil1D r1 `compare` SpecNNil1D r2 = r1 `compare` r2
  SpecNNil2D r1 c1 `compare` SpecNNil2D r2 c2 = compare (r1, c1) (r2, c2)
  SpecNNil3D r1 c1 d1 `compare` SpecNNil3D r2 c2 d2 = compare (r1, c1, d1) (r2, c2, d2)
  SpecNCons l1 r1 `compare` SpecNCons l2 r2 = compare (l1, r1) (l2, r2)
  SpecNetLayer spec1 `compare` SpecNetLayer spec2
    | T.typeOf spec1 == T.typeOf spec2 = compare (Just spec1) (cast spec2)
    | otherwise = LT
  x `compare` y = compare x y

instance NFData SpecNet where
  rnf (SpecNNil1D r1) = rnf r1 
  rnf (SpecNNil2D r1 c1) = rnf r1 `seq` rnf c1
  rnf (SpecNNil3D r1 c1 d1) = rnf r1 `seq` rnf c1 `seq` rnf d1
  rnf (SpecNCons l1 r1) = rnf l1 `seq` rnf r1
  rnf (SpecNetLayer spec1) = rnf spec1

instance Serialize SpecNet where
  put (SpecNNil1D r1)       = put (0 :: Int) >> put r1 
  put (SpecNNil2D r1 c1)    = put (1 :: Int) >> put r1 >> put c1
  put (SpecNNil3D r1 c1 d1) = put (2 :: Int) >> put r1 >> put c1 >> put d1
  put (SpecNCons l1 r1)     = put (3 :: Int) >> put l1 >> put r1
  put (SpecNetLayer spec1)  = put (4 :: Int) >> put (show spec1)
  get = do
    (nr :: Int) <- get
    case nr of
      0 -> SpecNNil1D <$> get
      1 -> SpecNNil2D <$> get <*> get
      2 -> SpecNNil3D <$> get <*> get <*> get
      3 -> SpecNCons <$> get <*> get
      4 -> do
        error "not yet implemented! Need to parse and initiate layers in Serialize instance of SpecNet!!!"
        -- layer <- read <$> get
        -- return $ SpecNetLayer layer
      _ -> error "unexpected input in Serialize instance of DynamicNetwork"

instance Show SpecNet where
  show (SpecNNil1D o) = "SpecNNil1D " ++ show o
  show (SpecNNil2D r c) = "SpecNNil2D " ++ show r ++ "x" ++ show c
  show (SpecNNil3D x y z) = "SpecNNil3D " ++ show x ++ "x" ++ show y ++ "x" ++ show z
  show (SpecNCons layer rest) = show layer ++ " :=> " ++ show rest
  show (SpecNetLayer layer) = show layer


---------------------------------------- 

-- Network instances 

instance (KnownNat nr) => FromDynamicLayer (Network '[] '[ 'D1 nr ]) where
  fromDynamicLayer _ = SpecNNil1D (natVal (Proxy :: Proxy nr))

instance (KnownNat rows, KnownNat cols) => FromDynamicLayer (Network '[] '[ 'D2 rows cols ]) where
  fromDynamicLayer _ = SpecNNil2D (natVal (Proxy :: Proxy rows)) (natVal (Proxy :: Proxy cols))

instance (KnownNat rows, KnownNat cols, KnownNat depth, KnownNat (rows GHC.TypeLits.* depth)) => FromDynamicLayer (Network '[] '[ 'D3 rows cols depth ]) where
  fromDynamicLayer _ = SpecNNil3D (natVal (Proxy :: Proxy rows)) (natVal (Proxy :: Proxy cols)) (natVal (Proxy :: Proxy depth))

instance (FromDynamicLayer x, FromDynamicLayer (Network xs (h : rs)), SingI i, SingI h, Layer x i h) => FromDynamicLayer (Network (x ': xs) (i ': h ': rs)) where
  fromDynamicLayer ((x :: x) :~> (xs :: Network xs (h ': rs))) = SpecNCons (fromDynamicLayer x) (fromDynamicLayer xs) 

instance ToDynamicLayer SpecNet where
  toDynamicLayer _ _ (SpecNNil1D nrOut) =
    reifyNat nrOut $ \(_ :: (KnownNat o') => Proxy o') ->
      case (unsafeCoerce (Dict :: Dict ()) :: Dict (layers ~ '[]), unsafeCoerce (Dict :: Dict ()) :: Dict (shapes ~ '[ 'D1 o'])) of
        (Dict, Dict) -> return $ SpecNetwork (NNil :: Network '[] '[ 'D1 o'])

  toDynamicLayer _ _ (SpecNNil2D rows cols) =
    reifyNat rows $ \(_ :: (KnownNat r) => Proxy r) -> reifyNat cols $ \(_ :: (KnownNat c) => Proxy c) -> 
      case (unsafeCoerce (Dict :: Dict ()) :: Dict (layers ~ '[]), unsafeCoerce (Dict :: Dict ()) :: Dict (shapes ~ '[ 'D2 r c ])) of
        (Dict, Dict) -> return $ SpecNetwork (NNil :: Network '[] '[ 'D2 r c])

  toDynamicLayer _ _ (SpecNNil3D rows cols depth) = 
    reifyNat rows $ \(_ :: (KnownNat r) => Proxy r) -> reifyNat cols $ \(_ :: (KnownNat c) => Proxy c) -> reifyNat depth $ \(_ :: (KnownNat d) => Proxy d) ->
      case ( unsafeCoerce (Dict :: Dict ()) :: Dict (layers ~ '[])
           , unsafeCoerce (Dict :: Dict ()) :: Dict (shapes ~ '[ 'D3 r c d ])
           , unsafeCoerce (Dict :: Dict ()) :: Dict (KnownNat (r GHC.TypeLits.* d))) of
        (Dict, Dict, Dict) -> return $ SpecNetwork (NNil :: Network '[] '[ 'D3 r c d])

  toDynamicLayer wInit gen (SpecNetLayer layer) = toDynamicLayer wInit gen layer

  toDynamicLayer wInit gen (SpecNCons left right) = do
    restNet <- toDynamicLayer wInit gen right
    case restNet of
      SpecLayer{} -> error "Expected a network, but received a layer in Specification"
      SpecNetwork (xs :: Network restLayers restShapes) -> 
        case (sing :: Sing restShapes) of
          SNil -> error "unexpected empty network (shapes where SNil) in Specification"
          (SCons (h :: Sing h) (_ :: Sing hs)) ->
            withSingI h $ do
              layer <- toDynamicLayer wInit gen left
              case layer of 
                SpecLayer (x :: xType) sInShape sOutShape -> 
                  case (sInShape, sOutShape) of 
                    (SomeSing (_ :: Sing inShape), SomeSing (_ :: Sing outShape)) -> 
                      case ( unsafeCoerce (Dict :: Dict ()) :: Dict (outShape ~ h)
                           , unsafeCoerce (Dict :: Dict ()) :: Dict (Layer xType inShape outShape)
                           , unsafeCoerce (Dict :: Dict ()) :: Dict (SingI inShape)) of
                        (Dict, Dict, Dict) -> return $ SpecNetwork (x :~> xs :: Network (xType ': restLayers) (inShape ': h ': hs))
                SpecNetwork (x :: Network xLayers xShapes) -> 
                  case (sing :: Sing xShapes) of 
                    SNil -> error "unexpected empty network (SpecNNil) as layer in specification. Cannot proceed."
                    SCons (_ :: Sing i) (_ :: Sing xHs) ->
                      case ( unsafeCoerce (Dict :: Dict ()) :: Dict (Last xHs ~ h)
                           , unsafeCoerce (Dict :: Dict ()) :: Dict (Layer (Network xLayers xShapes) i (Last xHs))) of
                        (Dict, Dict) -> return $ SpecNetwork (x :~> xs :: Network (Network xLayers xShapes ': restLayers) (i ': h ': hs))

----------------------------------------

-- Some Convenience functions

-- | Create a network according to the given specification. See @DynamicNetwork@. This version uses UniformInit and the system random number generator.
-- networkFromSpecification :: forall layers shapes . (DynamicLayer (Network layers shapes) (Head shapes) (Last shapes)) => Specification -> IO SpecNetwork
-- networkFromSpecification :: forall (pxLayers :: [Type]) (pxShapes :: [Shape]) (layersOut :: [Type]) (shapesOut :: [Shape]) . Specification -> IO SpecNetwork
networkFromSpecification :: SpecNet -> IO SpecNetwork
networkFromSpecification spec = withSystemRandom . asGenST $ \gen -> toDynamicLayer UniformInit gen spec


-- | Combine specifications together. This is (:~>) for specifications. This is simply SpecNCons as operator.
(|=>) :: SpecNet -> SpecNet -> SpecNet
l |=> r = SpecNCons l r
infixr 5 |=> 

-- | 1D Output layer for specification. Requieres output size.
specNil1D :: Integer -> SpecNet
specNil1D  = SpecNNil1D

-- | 2D Output layer for specification. Requieres output rows and cols.
specNil2D :: Integer -> Integer -> SpecNet
specNil2D = SpecNNil2D 

-- | 3D Output layer for specification. Requieres output sizes.
specNil3D :: Integer -> Integer -> Integer -> SpecNet
specNil3D = SpecNNil3D 


