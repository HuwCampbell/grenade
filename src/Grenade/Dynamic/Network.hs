{-# LANGUAGE CPP                   #-}
{-# LANGUAGE ConstraintKinds       #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE DeriveAnyClass        #-}
{-# LANGUAGE DeriveGeneric         #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE InstanceSigs          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedStrings     #-}
{-# LANGUAGE PolyKinds             #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE UndecidableInstances  #-}
{-|
Module      : Grenade.Dynamic.Network
Description : Convienence functions for dynamic networks.
Copyright   : (c) Manuel Schneckenreither, 2016-2020
License     : BSD2
Stability   : experimental
-}
module Grenade.Dynamic.Network
  ( SpecConcreteNetwork (..)
  -- , withConcreteNetwork
  , networkFromSpecification
  , networkFromSpecificationWith
  , networkFromSpecificationGenericWith
  , networkToSpecification
  -- Convenience functions for creating dynamic network specifications:
  , (|=>)
  , specNil1D
  , specNil2D
  , specNil3D
  , specNil
  -- Helpers
  , tripleFromSomeShape
  ) where

import           Control.DeepSeq
import           Data.Constraint                  (Dict (..))
import           Data.Default
import           Data.List.Singletons
import           Data.Serialize
import           Data.Singletons
import           Data.Typeable                    (Typeable)
import           GHC.TypeLits
import           GHC.TypeLits.Singletons          hiding (natVal)
import           System.Random.MWC
import           Unsafe.Coerce                    (unsafeCoerce)

import           Grenade.Core.Layer
import           Grenade.Core.Network
import           Grenade.Core.NetworkInitSettings
import           Grenade.Core.Shape
import           Grenade.Dynamic.Specification
import           Grenade.Layers.Internal.CUDA     (setCudaTriggerSize)

----------------------------------------
-- Some Convenience functions

-- | Instances all networks implement.
type GeneralConcreteNetworkInstances layers shapes
   = ( FromDynamicLayer (Network layers shapes)
     , FoldableGradient (Gradients layers)
     , GNum (Gradients layers)
     , GNum (Network layers shapes)
     , Layer (Network layers shapes) (Head shapes) (Last shapes)
     , NFData (Gradients layers)
     , NFData (Network layers shapes)
     , NFData (Tapes layers shapes)
     , RandomLayer (Network layers shapes)
     , Serialize (Network layers shapes)
     , Serialize (Gradients layers)
     , Show (Network layers shapes)
     , SingI (Head shapes)
     , SingI (Last shapes)
     , SingI shapes
     , Typeable layers
     , Typeable shapes
     , CreatableNetwork layers shapes
     )


-- | This is the result type when calling @networkFromSpecification@. It specifies the input and output type. For a generic version (where input and output type are unknown) see @SpecNetwork@ and
-- @networkFromSpecificationGeneric@.
data SpecConcreteNetwork where
  SpecConcreteNetwork1D1D
    :: ( GeneralConcreteNetworkInstances layers shapes, Head shapes ~ 'D1 i1, Last shapes ~ 'D1 o1, KnownNat i1, KnownNat o1)
    => !(Network layers shapes) -> SpecConcreteNetwork
  SpecConcreteNetwork1D2D
    :: ( GeneralConcreteNetworkInstances layers shapes, Head shapes ~ 'D1 i1, Last shapes ~ 'D2 o1 o2, KnownNat i1, KnownNat o1, KnownNat o2)
    => !(Network layers shapes) -> SpecConcreteNetwork
  SpecConcreteNetwork1D3D
    :: ( GeneralConcreteNetworkInstances layers shapes, Head shapes ~ 'D1 i1, Last shapes ~ 'D3 o1 o2 o3, KnownNat i1, KnownNat o1, KnownNat o2, KnownNat o3)
    => !(Network layers shapes) -> SpecConcreteNetwork
  SpecConcreteNetwork2D1D
    :: ( GeneralConcreteNetworkInstances layers shapes, Head shapes ~ 'D2 i1 i2, Last shapes ~ 'D1 o, KnownNat i1, KnownNat i2, KnownNat o)
    => !(Network layers shapes) -> SpecConcreteNetwork
  SpecConcreteNetwork2D2D
    :: ( GeneralConcreteNetworkInstances layers shapes, Head shapes ~ 'D2 i1 i2, Last shapes ~ 'D2 o1 o2, KnownNat i1, KnownNat i2, KnownNat o1, KnownNat o2)
    => !(Network layers shapes) -> SpecConcreteNetwork
  SpecConcreteNetwork2D3D
    :: ( GeneralConcreteNetworkInstances layers shapes, Head shapes ~ 'D2 i1 i2, Last shapes ~ 'D3 o1 o2 o3, KnownNat i1, KnownNat i2, KnownNat o1, KnownNat o2, KnownNat o3)
    => !(Network layers shapes) -> SpecConcreteNetwork
  SpecConcreteNetwork3D1D
    :: ( GeneralConcreteNetworkInstances layers shapes, Head shapes ~ 'D3 i1 i2 i3, Last shapes ~ 'D1 o, KnownNat i1, KnownNat i2, KnownNat i3, KnownNat o)
    => !(Network layers shapes) -> SpecConcreteNetwork
  SpecConcreteNetwork3D2D
    :: ( GeneralConcreteNetworkInstances layers shapes, Head shapes ~ 'D3 i1 i2 i3, Last shapes ~ 'D2 o1 o2, KnownNat i1, KnownNat i2, KnownNat i3, KnownNat o1, KnownNat o2)
    => !(Network layers shapes) -> SpecConcreteNetwork
  SpecConcreteNetwork3D3D
    :: ( GeneralConcreteNetworkInstances layers shapes, Head shapes ~ 'D3 i1 i2 i3, Last shapes ~ 'D3 o1 o2 o3, KnownNat i1, KnownNat i2, KnownNat i3, KnownNat o1, KnownNat o2, KnownNat o3)
    => !(Network layers shapes) -> SpecConcreteNetwork


-- | Create a network according to the given specification. See @DynamicNetwork@. This version uses UniformInit and the system random number generator. WARNING: This also allows to build unsafe
-- networks where input and output layers do not match! Thus use with care!
networkFromSpecification :: SpecNet -> IO SpecConcreteNetwork
networkFromSpecification = networkFromSpecificationWith def


-- | Create a network according to the given specification. See @DynamicNetwork@. This version uses UniformInit and the system random number generator. WARNING: This also allows to build unsafe
-- networks where input and output layers do not match! Thus use with care!
networkFromSpecificationWith :: NetworkInitSettings -> SpecNet -> IO SpecConcreteNetwork
networkFromSpecificationWith wInit spec = do
  SpecNetwork (net :: Network layers shapes) <- withSystemRandom . asGenST $ \gen -> toDynamicLayer wInit gen spec
  setCudaTriggerSize (gpuTriggerSize wInit)
  case (sing :: Sing (Head shapes), sing :: Sing (Last shapes), unsafeCoerce (Dict :: Dict ()) :: Dict ()) of
    (inp :: Sing (Head shapes), out :: Sing (Last shapes), Dict) ->
      withSingI inp $
      withSingI out $
      case (inp, out) of
        (D1Sing SNat, D1Sing SNat)                     -> return $ SpecConcreteNetwork1D1D net
        (D1Sing SNat, D2Sing SNat SNat)                -> return $ SpecConcreteNetwork1D2D net
        (D1Sing SNat, D3Sing SNat SNat SNat)           -> return $ SpecConcreteNetwork1D3D net
        (D2Sing SNat SNat, D1Sing SNat)                -> return $ SpecConcreteNetwork2D1D net
        (D2Sing SNat SNat, D2Sing SNat SNat)           -> return $ SpecConcreteNetwork2D2D net
        (D2Sing SNat SNat, D3Sing SNat SNat SNat)      -> return $ SpecConcreteNetwork2D3D net
        (D3Sing SNat SNat SNat, D1Sing SNat)           -> return $ SpecConcreteNetwork3D1D net
        (D3Sing SNat SNat SNat, D2Sing SNat SNat)      -> return $ SpecConcreteNetwork3D2D net
        (D3Sing SNat SNat SNat, D3Sing SNat SNat SNat) -> return $ SpecConcreteNetwork3D3D net

-- | Create a network according to the given specification. See @DynamicNetwork@. This version uses UniformInit and the system random number generator. WARNING: This also allows to build unsafe
-- networks where input and output layers do not match! Thus use with care! Furthermore, if you need to specify the actual I/O types, see @networkFromSpecificationWith@ for implementation details!
networkFromSpecificationGenericWith :: NetworkInitSettings -> SpecNet -> IO SpecNetwork
networkFromSpecificationGenericWith wInit spec = withSystemRandom . asGenST $ \gen -> setCudaTriggerSize (gpuTriggerSize wInit) >> toDynamicLayer wInit gen spec


-- | Create a network according to the given specification. See @DynamicNetwork@. This version uses UniformInit and the system random number generator.
networkToSpecification :: forall layers shapes . (SingI (Head shapes), SingI (Last shapes), FromDynamicLayer (Network layers shapes)) => Network layers shapes -> SpecNet
networkToSpecification = fromDynamicLayer (SomeSing (sing :: Sing (Head shapes))) (SomeSing (sing :: Sing (Last shapes)))


-- | Combine specifications together. This is (:~>) for specifications. This is simply SpecNCons as operator. WARNING: This also allows to build unsafe networks where input and output layers do not
-- match! Thus use with care!
(|=>) :: SpecNet -> SpecNet -> SpecNet
l |=> r = SpecNCons l r
infixr 5 |=>

-- | 1D Output layer for specification. Requieres output size.
specNil1D :: Integer -> SpecNet
specNil1D  = SpecNNil1D

-- | 2D Output layer for specification. Requieres output rows and cols.
specNil2D :: (Integer, Integer) -> SpecNet
specNil2D = uncurry SpecNNil2D

-- | 3D Output layer for specification. Requieres output sizes.
specNil3D :: (Integer, Integer, Integer) -> SpecNet
specNil3D (rows, cols, depth) = SpecNNil3D rows cols depth

specNil :: (Integer, Integer, Integer) -> SpecNet
specNil dimensions =
  case dimensions of
    (rows, 1, 1)        -> specNil1D rows
    (rows, cols, 1)     -> specNil2D (rows, cols)
    (rows, cols, depth) -> specNil3D (rows, cols, depth)


-- | Helper functions to convert a given shape into a triple, where nonused dimensions are set to 0.
tripleFromSomeShape :: SomeSing Shape -> (Integer, Integer, Integer)
tripleFromSomeShape someShape =
  case someShape of
    SomeSing (shape :: Sing shape) ->
      withSingI shape $
      case shape of
        D1Sing r@SNat               -> (natVal r, 1, 1)
        D2Sing r@SNat c@SNat        -> (natVal r, natVal c, 1)
        D3Sing r@SNat c@SNat d@SNat -> (natVal r, natVal c, natVal d)
