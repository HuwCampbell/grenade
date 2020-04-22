{-# LANGUAGE CPP                       #-}
{-# LANGUAGE DataKinds                 #-}
{-# LANGUAGE DeriveAnyClass            #-}
{-# LANGUAGE DeriveGeneric             #-}
{-# LANGUAGE ExistentialQuantification #-}
{-# LANGUAGE FlexibleContexts          #-}
{-# LANGUAGE FlexibleInstances         #-}
{-# LANGUAGE GADTs                     #-}
{-# LANGUAGE KindSignatures            #-}
{-# LANGUAGE RankNTypes                #-}
{-# LANGUAGE ScopedTypeVariables       #-}
{-# LANGUAGE TypeOperators             #-}
{-|
Module      : Grenade.Dynamic.Specification
Description : Specifying network architectures for dynamically creating nets.
Copyright   : (c) Manuel Schneckenreither, 2020
License     : BSD2
Stability   : experimental
-}
module Grenade.Dynamic.Specification
  ( FromDynamicLayer (..)
  , ToDynamicLayer (..)
  , SpecNetwork (..)
  , SpecNet (..)
  -- Layer specific data types
  , Dimensions
  , SpecFullyConnected(..)
  , SpecConvolution(..)
  , SpecDeconvolution(..)
  , SpecDropout(..)
  , SpecElu(..)
  , SpecLogit(..)
  , SpecRelu(..)
  , SpecReshape(..)
  , SpecSinusoid(..)
  , SpecSoftmax(..)
  , SpecTanh(..)
  , SpecTrivial(..)
  ) where

import           Control.DeepSeq
import           Data.Constraint               (Dict (..))
import           Data.Proxy
import           Data.Reflection               (reifyNat)
import           Data.Serialize
import           Data.Singletons
import           Data.Singletons.Prelude.List
import           Data.Typeable                 as T
import           GHC.Generics
import           GHC.TypeLits
import           Unsafe.Coerce                 (unsafeCoerce)
#if MIN_VERSION_base(4,9,0)
import           Data.Kind                     (Type)
#endif
#ifndef FLYCHECK
import {-# SOURCE #-} Grenade.Layers.Convolution    ()
import {-# SOURCE #-} Grenade.Layers.Deconvolution  ()
import {-# SOURCE #-} Grenade.Layers.Dropout        ()
import {-# SOURCE #-} Grenade.Layers.Elu            ()
import {-# SOURCE #-} Grenade.Layers.FullyConnected ()
import {-# SOURCE #-} Grenade.Layers.Logit          ()
import {-# SOURCE #-} Grenade.Layers.Relu           ()
import {-# SOURCE #-} Grenade.Layers.Reshape        ()
import {-# SOURCE #-} Grenade.Layers.Sinusoid       ()
import {-# SOURCE #-} Grenade.Layers.Softmax        ()
import {-# SOURCE #-} Grenade.Layers.Tanh           ()
import {-# SOURCE #-} Grenade.Layers.Trivial        ()
#endif


import           Control.Monad.Primitive       (PrimBase, PrimState)
import           Grenade.Core
import           Grenade.Types
import           System.Random.MWC             (Gen)


-- | Create a runtime dynamic specification of a network. Dynamic layers (and networks), for storing and restoring specific network structures (e.g. in saving the network structures to a DB and
-- restoring it from there) or simply generating them at runtime. This does not store the weights and biases! They have to be handled separately (see Serialize)!
class FromDynamicLayer x where
  fromDynamicLayer :: SomeSing Shape -> SomeSing Shape -> x -> SpecNet

-- | Class for generating layers from a specification. See the function @networkFromSpecification@.
class (Show spec) => ToDynamicLayer spec where
  toDynamicLayer :: (PrimBase m) => WeightInitMethod -> Gen (PrimState m) -> spec -> m SpecNetwork


----------------------------------------
-- Return value of toDynamicLayer

-- | Specification of a network or layer.
data SpecNetwork :: Type where
  SpecNetwork
    :: (SingI shapes, SingI (Head shapes), SingI (Last shapes), Show (Network layers shapes), FromDynamicLayer (Network layers shapes), NFData (Network layers shapes)
       , Layer (Network layers shapes) (Head shapes) (Last shapes), RandomLayer (Network layers shapes), Serialize (Network layers shapes), GNum (Network layers shapes)
       , NFData (Tapes layers shapes), GNum (Gradients layers), Typeable layers, Typeable shapes, Typeable (Head shapes))
    => !(Network layers shapes)
    -> SpecNetwork
  SpecLayer :: (FromDynamicLayer x, RandomLayer x, Serialize x, Typeable x, Typeable i, NFData (Tape x i o), GNum (Gradient x), GNum x, NFData x, Show x, Layer x i o) => !x -> !(Sing i) -> !(Sing o) -> SpecNetwork

instance Show SpecNetwork where
  show (SpecNetwork net) = show net
  show (SpecLayer x _ _) = show x

instance FromDynamicLayer SpecNetwork where
  fromDynamicLayer inp out (SpecNetwork net) = fromDynamicLayer inp out net
  fromDynamicLayer inp out (SpecLayer x _ _) = fromDynamicLayer inp out x


-- | Data structure for holding specifications for networks. Networks can be built dynamically with @toDynamicLayer@. Further see the functions @|=>@, @specNil1D@, @specNil2D@, @specNil3D@, and
-- possibly any other layer implementation of @ToDynamicLayer@ for building specifications. See functions @networkFromSpecification@ to create a network from a specification and
-- @networkToSpecification@ to create a specification from a network.
data SpecNet
  = SpecNNil1D !Integer                   -- ^ 1D network output
  | SpecNNil2D !Integer !Integer          -- ^ 2D network output
  | SpecNNil3D !Integer !Integer !Integer -- ^ 3D network output
  | SpecNCons !SpecNet !SpecNet           -- ^ x :~> xs, where x can also be a network
  | forall spec . (ToDynamicLayer spec, Typeable spec, Ord spec, Eq spec, Show spec, Serialize spec, NFData spec) => SpecNetLayer !spec -- ^ Specification of a layer

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
  rnf (SpecNNil1D r1)       = rnf r1
  rnf (SpecNNil2D r1 c1)    = rnf r1 `seq` rnf c1
  rnf (SpecNNil3D r1 c1 d1) = rnf r1 `seq` rnf c1 `seq` rnf d1
  rnf (SpecNCons l1 r1)     = rnf l1 `seq` rnf r1
  rnf (SpecNetLayer spec1)  = rnf spec1

instance Serialize SpecNet where
  put (SpecNNil1D r1)       = put (0 :: Int) >> put r1
  put (SpecNNil2D r1 c1)    = put (1 :: Int) >> put r1 >> put c1
  put (SpecNNil3D r1 c1 d1) = put (2 :: Int) >> put r1 >> put c1 >> put d1
  put (SpecNCons l1 r1)     = put (3 :: Int) >> put l1 >> put r1
  put (SpecNetLayer spec1)  = put (4 :: Int) >> put (show $ typeOf spec1) >> put spec1
  get = do
    (nr :: Int) <- get
    case nr of
      0 -> SpecNNil1D <$> get
      1 -> SpecNNil2D <$> get <*> get
      2 -> SpecNNil3D <$> get <*> get <*> get
      3 -> SpecNCons <$> get <*> get
      4 -> get >>= parseSpecDataConstructor
      _ -> error "unexpected input in Serialize instance of DynamicNetwork"

instance Show SpecNet where
  show (SpecNNil1D o) = "SpecNNil1D " ++ show o
  show (SpecNNil2D r c) = "SpecNNil2D " ++ show r ++ "x" ++ show c
  show (SpecNNil3D x y z) = "SpecNNil3D " ++ show x ++ "x" ++ show y ++ "x" ++ show z
  show (SpecNCons layer rest) = case layer of
    SpecNCons{} -> "[ " ++ show layer ++ " ] :=> " ++ show rest
    _           -> show layer ++ " :=> " ++ show rest
  show (SpecNetLayer layer) = show layer


-- | Function to parse the known implemented types. Can only deserialise these!
parseSpecDataConstructor :: String -> Get SpecNet
#ifdef FLYCHECK
parseSpecDataConstructor = error "disable -DFLYCHECK while compiling!"
#else
parseSpecDataConstructor repStr
  | repStr == show (typeRep (Proxy :: Proxy SpecFullyConnected)) = SpecNetLayer <$> (get :: Get SpecFullyConnected)
  | repStr == show (typeRep (Proxy :: Proxy SpecConvolution))    = SpecNetLayer <$> (get :: Get SpecConvolution)
  | repStr == show (typeRep (Proxy :: Proxy SpecDeconvolution))  = SpecNetLayer <$> (get :: Get SpecDeconvolution)
  | repStr == show (typeRep (Proxy :: Proxy SpecDropout))        = SpecNetLayer <$> (get :: Get SpecDropout)
  | repStr == show (typeRep (Proxy :: Proxy SpecElu))            = SpecNetLayer <$> (get :: Get SpecElu)
  | repStr == show (typeRep (Proxy :: Proxy SpecLogit))          = SpecNetLayer <$> (get :: Get SpecLogit)
  | repStr == show (typeRep (Proxy :: Proxy SpecRelu))           = SpecNetLayer <$> (get :: Get SpecRelu)
  | repStr == show (typeRep (Proxy :: Proxy SpecReshape))        = SpecNetLayer <$> (get :: Get SpecReshape)
  | repStr == show (typeRep (Proxy :: Proxy SpecSinusoid))       = SpecNetLayer <$> (get :: Get SpecSinusoid)
  | repStr == show (typeRep (Proxy :: Proxy SpecSoftmax))        = SpecNetLayer <$> (get :: Get SpecSoftmax)
  | repStr == show (typeRep (Proxy :: Proxy SpecTanh))           = SpecNetLayer <$> (get :: Get SpecTanh)
  | repStr == show (typeRep (Proxy :: Proxy SpecTrivial))        = SpecNetLayer <$> (get :: Get SpecTrivial)
  | repStr == show (typeRep (Proxy :: Proxy SpecNet))            = SpecNetLayer <$> (get :: Get SpecNet)
  | otherwise = error $ "unexpected input parseSpecDataConstructor: " ++ repStr
#endif


----------------------------------------

-- Network instances

instance (KnownNat rows) => FromDynamicLayer (Network '[] '[ 'D1 rows ]) where
  fromDynamicLayer _ _ _ = SpecNNil1D nr
    where nr = natVal (Proxy :: Proxy rows)
instance (KnownNat rows, KnownNat cols) => FromDynamicLayer (Network '[] '[ 'D2 rows cols ]) where
  fromDynamicLayer _ _ _ = SpecNNil2D (natVal (Proxy :: Proxy rows)) (natVal (Proxy :: Proxy cols))

instance (KnownNat rows, KnownNat cols, KnownNat depth) => FromDynamicLayer (Network '[] '[ 'D3 rows cols depth ]) where
  fromDynamicLayer _ _ _ = SpecNNil3D (natVal (Proxy :: Proxy rows)) (natVal (Proxy :: Proxy cols)) (natVal (Proxy :: Proxy depth))

instance (FromDynamicLayer x, FromDynamicLayer (Network xs (h : rs)), SingI h) => FromDynamicLayer (Network (x ': xs) (i ': h ': rs)) where
  fromDynamicLayer inp out ((x :: x) :~> (xs :: Network xs (h ': rs))) = SpecNCons (fromDynamicLayer inp hShape x) (fromDynamicLayer hShape out xs)
    where
      hShape = SomeSing (sing :: Sing h)

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
          (SCons (h :: Sing h) (hs :: Sing hs)) -> withSingI h $ withSingI hs $ do
            layer <- toDynamicLayer wInit gen left
            case layer of
              SpecLayer (x :: xType) singIn singOut ->
                case (singIn, singOut) of
                  (pxIn :: Sing inShape, pxOut :: Sing outShape) ->
                    withSingI pxIn $ withSingI h $ withSingI pxOut $
                     case ( unsafeCoerce (Dict :: Dict ()) :: Dict (outShape ~ h)
                          , unsafeCoerce (Dict :: Dict ()) :: Dict (CreatableNetwork (xType ': restLayers) (inShape ': restShapes))) of
                       (Dict, Dict) -> return $ SpecNetwork (x :~> xs :: Network (xType ': restLayers) (inShape ': restShapes))
              SpecNetwork (x :: Network xLayers xShapes) ->
                case (sing :: Sing xShapes) of
                  SNil -> error "unexpected empty network (SpecNNil) as layer in specification. Cannot proceed."
                  SCons (i :: Sing i) (xHs :: Sing xHs) -> withSingI i $ withSingI xHs $
                    case ( unsafeCoerce (Dict :: Dict ()) :: Dict (Head restShapes ~ Last xShapes)
                         , unsafeCoerce (Dict :: Dict ()) :: Dict (CreatableNetwork (Network xLayers xShapes : restLayers) (i ': restShapes))) of
                      (Dict, Dict) -> return $ SpecNetwork (x :~> xs :: Network (Network xLayers xShapes ': restLayers) (i ': restShapes))

-- Data structures stances for Layers (needs to be defined here)

type Dimensions = (Integer, Integer, Integer)

-- | Data Structure to save a fully connected layer. Saves number of input and output nodes.
data SpecFullyConnected = SpecFullyConnected !Integer !Integer
  deriving (Show, Eq, Ord, Serialize, Generic, NFData)

-- data SpecConcat = SpecConcat SpecNet SpecNet
--   deriving (Show, Eq, Ord, Serialize, Generic, NFData)

-- | Specifiation of a convolutional layer. Saves input and channels, filters, kernelRows, kernelColumns, strideRows, strideColumns, and kernelFlattened.
data SpecConvolution =
  SpecConvolution !Dimensions !Integer !Integer !Integer !Integer !Integer !Integer
  deriving (Show, Read, Eq, Ord, Serialize, Generic, NFData)

-- | Specifiation of a deconvolutional layer. Saves input and channels, filters, kernelRows, kernelColumns, strideRows, strideColumns, and kernelFlattened.
data SpecDeconvolution =
  SpecDeconvolution !Dimensions !Integer !Integer !Integer !Integer !Integer !Integer
  deriving (Show, Read, Eq, Ord, Serialize, Generic, NFData)

-- | Specification of a dropout filter. Saves number of input nodes, ratio and maybe a seed.
data SpecDropout = SpecDropout !Integer !RealNum !(Maybe Int)
  deriving (Show, Read, Eq, Ord, Serialize, Generic, NFData)

-- | Specification of Elu, saves input dimensions as triple, where every element >= 1.
newtype SpecElu = SpecElu Dimensions
  deriving (Show, Read, Eq, Ord, Serialize, Generic, NFData)

-- | Specification of Logit, saves input dimensions as triple, where every element >= 1.
newtype SpecLogit = SpecLogit Dimensions
  deriving (Show, Read, Eq, Ord, Serialize, Generic, NFData)

-- | Specification of Relu, saves input dimensions as triple, where every element >= 1.
newtype SpecRelu = SpecRelu Dimensions
  deriving (Show, Read, Eq, Ord, Serialize, Generic, NFData)

-- | Specification of a reshape layer, saves input and output dimensions, where every element >= 1.
data SpecReshape = SpecReshape !Dimensions !Dimensions
  deriving (Show, Read, Eq, Ord, Serialize, Generic, NFData)

-- | Specification of Sinusoid, saves input dimensions as triple, where every element >= 1.
newtype SpecSinusoid = SpecSinusoid Dimensions
  deriving (Show, Read, Eq, Ord, Serialize, Generic, NFData)

-- | Specification of Softmax, saves input dimensions as triple, where every element >= 1.
newtype SpecSoftmax = SpecSoftmax Integer
  deriving (Show, Read, Eq, Ord, Serialize, Generic, NFData)

-- | Specification of Tanh, saves input dimensions as triple, where every element >= 1.
newtype SpecTanh = SpecTanh Dimensions
  deriving (Show, Read, Eq, Ord, Serialize, Generic, NFData)

-- | Specification of Trivial, saves input dimensions as triple, where every element >= 1.
newtype SpecTrivial = SpecTrivial Dimensions
  deriving (Show, Read, Eq, Ord, Serialize, Generic, NFData)


