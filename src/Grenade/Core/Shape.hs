{-# LANGUAGE BangPatterns         #-}
{-# LANGUAGE CPP                  #-}
{-# LANGUAGE DataKinds            #-}
{-# LANGUAGE FlexibleContexts     #-}
{-# LANGUAGE GADTs                #-}
{-# LANGUAGE RankNTypes           #-}
{-# LANGUAGE ScopedTypeVariables  #-}
{-# LANGUAGE TypeFamilies         #-}
{-# LANGUAGE TypeOperators        #-}
{-# LANGUAGE UndecidableInstances #-}
{-|
Module      : Grenade.Core.Shape
Description : Dependently typed shapes of data which are passed between layers of a network
Copyright   : (c) Huw Campbell, 2016-2017
License     : BSD2
Stability   : experimental


-}
module Grenade.Core.Shape (
    S (..)
  , Shape (..)
#if MIN_VERSION_singletons(2,6,0)
  , SShape (..)
#else
  , Sing (..)
#endif

  , randomOfShape
  , fromStorable
  , fromStorableV
  ) where

#if MIN_VERSION_singletons(2,6,0)
import           Data.Kind                    (Type)
import           Data.Singletons.TypeLits     (SNat (..))
#endif

import           Control.DeepSeq              (NFData (..))
import           Data.Proxy
import           Data.Serialize
import           Data.Singletons
import           Data.Singletons.TypeLits
import           Data.Vector.Storable         (Vector)
import qualified Data.Vector.Storable         as V
import           GHC.TypeLits                 hiding (natVal)
import qualified Numeric.LinearAlgebra        as NLA
import           Numeric.LinearAlgebra.Static
import qualified Numeric.LinearAlgebra.Static as H
import           System.Random.MWC

import           Grenade.Types

-- | The current shapes we accept.
--   at the moment this is just one, two, and three dimensional
--   Vectors/Matricies.
--
--   These are only used with DataKinds, as Kind `Shape`, with Types 'D1, 'D2, 'D3.
data Shape
  = D1 Nat
  -- ^ One dimensional vector
  | D2 Nat Nat
  -- ^ Two dimensional matrix. Row, Column.
  | D3 Nat Nat Nat
  -- ^ Three dimensional matrix. Row, Column, Channels.

-- | Concrete data structures for a Shape.
--
--   All shapes are held in contiguous memory.
--   3D is held in a matrix (usually row oriented) which has height depth * rows.
data S (n :: Shape) where
  S1D :: ( KnownNat len )
      => R len
      -> S ('D1 len)

  S2D :: ( KnownNat rows, KnownNat columns )
      => L rows columns
      -> S ('D2 rows columns)

  S3D :: ( KnownNat rows
         , KnownNat columns
         , KnownNat depth
         , KnownNat (rows * depth))
      => L (rows * depth) columns
      -> S ('D3 rows columns depth)

  -- HMatrix instances
  S1DV :: (KnownNat len)
       => V.Vector RealNum
       -> S ('D1 len)

  -- HMatrix instances
  S2DV :: (KnownNat rows, KnownNat columns)
       => V.Vector RealNum -- ^ Vector in row major. Use Data.Matrix here?
       -> S ('D2 rows columns)

  -- HMatrix instances
  -- S3DH :: (KnownNat rows, KnownNat columns, KnownNat depth, KnownNat (rows * depth))
  --      => HBLAS.MDenseMatrix RealWorld 'HBLAS.Row Double
  --      -> S ('D3 rows columns depth)


instance Show (S n) where
  show (S1D x)  = "S1D " ++ show x
  show (S2D x)  = "S2D " ++ show x
  show (S3D x)  = "S3D " ++ show x
  show (S1DV x) = "S1DV " ++ show x
  show (S2DV x) = "S2DV " ++ show x
  -- show (S3DH x) = "S3DH " ++ show (unsafePerformIO $ V.freeze $ HBLAS._bufferDenMutMat x)

-- Singleton instances.
--
-- These could probably be derived with template haskell, but this seems
-- clear and makes adding the KnownNat constraints simple.
-- We can also keep our code TH free, which is great.
#if MIN_VERSION_singletons(2,6,0)
-- In singletons 2.6 Sing switched from a data family to a type family.
type instance Sing = SShape

data SShape :: Shape -> Type where
  D1Sing :: Sing a -> SShape ('D1 a)
  D2Sing :: Sing a -> Sing b -> SShape ('D2 a b)
  D3Sing :: KnownNat (a * c) => Sing a -> Sing b -> Sing c -> SShape ('D3 a b c)
#else
data instance Sing (n :: Shape) where
  D1Sing :: Sing a -> Sing ('D1 a)
  D2Sing :: Sing a -> Sing b -> Sing ('D2 a b)
  D3Sing :: KnownNat (a * c) => Sing a -> Sing b -> Sing c -> Sing ('D3 a b c)
#endif

instance KnownNat a => SingI ('D1 a) where
  sing = D1Sing sing
instance (KnownNat a, KnownNat b) => SingI ('D2 a b) where
  sing = D2Sing sing sing
instance (KnownNat a, KnownNat b, KnownNat c, KnownNat (a * c)) => SingI ('D3 a b c) where
  sing = D3Sing sing sing sing

instance SingI x => Num (S x) where
  (+) = n2 (+)
  (-) = n2 (-)
  (*) = n2 (*)
  abs = n1 abs
  signum = n1 signum
  fromInteger x = nk (fromInteger x)

instance SingI x => Fractional (S x) where
  (/) = n2 (/)
  recip = n1 recip
  fromRational x = nk (fromRational x)

instance SingI x => Floating (S x) where
  pi = nk pi
  exp = n1 exp
  log = n1 log
  sqrt = n1 sqrt
  (**) = n2 (**)
  logBase = n2 logBase
  sin = n1 sin
  cos = n1 cos
  tan = n1 tan
  asin = n1 asin
  acos = n1 acos
  atan = n1 atan
  sinh = n1 sinh
  cosh = n1 cosh
  tanh = n1 tanh
  asinh = n1 asinh
  acosh = n1 acosh
  atanh = n1 atanh

--
-- I haven't made shapes strict, as sometimes they're not needed
-- (the last input gradient back for instance)
--
instance NFData (S x) where
  rnf (S1D x)   = rnf x
  rnf (S2D x)   = rnf x
  rnf (S3D x)   = rnf x
  rnf (S1DV !v) = rnf v
  rnf (S2DV !v) = rnf v
  -- rnf (S3DH !_) = ()

-- | Generate random data of the desired shape
randomOfShape :: forall x . (SingI x) => IO (S x)
randomOfShape = do
  seed :: Int <- withSystemRandom . asGenST $ \gen -> uniform gen
  return $ case (sing :: Sing x) of
    D1Sing SNat ->
        S1D (randomVector seed Uniform * 2 - 1)

    D2Sing SNat SNat ->
        S2D (uniformSample seed (-1) 1)

    D3Sing SNat SNat SNat ->
        S3D (uniformSample seed (-1) 1)

-- | Generate a shape from a Storable Vector.
--
--   Returns Nothing if the vector is of the wrong size.
fromStorable :: forall x. SingI x => Vector RealNum -> Maybe (S x)
fromStorable xs =
  case (sing :: Sing x) of
    D1Sing SNat           -> S1D <$> H.create xs
    D2Sing SNat SNat      -> S2D <$> mkL xs
    D3Sing SNat SNat SNat -> S3D <$> mkL xs
  where
    mkL ::
         forall rows columns. (KnownNat rows, KnownNat columns)
      => Vector RealNum
      -> Maybe (L rows columns)
    mkL v =
      let rows = fromIntegral $ natVal (Proxy :: Proxy rows)
          columns = fromIntegral $ natVal (Proxy :: Proxy columns)
       in if rows * columns == V.length v
            then H.create $ NLA.reshape columns v
            else Nothing

fromStorableV :: forall x. SingI x => Vector RealNum -> S x
fromStorableV v =
  case (sing :: Sing x) of
    D1Sing SNat           -> S1DV v
    D2Sing SNat SNat      -> S2DV v
    D3Sing SNat SNat SNat -> error "unexpected case in fromStorableS"

instance SingI x => Serialize (S x) where
  put i =
    case i of
      S1D x -> put (1 :: Int) >> (putListOf put . NLA.toList . H.extract $ x)
      S2D x -> put (1 :: Int) >> (putListOf put . NLA.toList . NLA.flatten . H.extract $ x)
      S3D x -> put (1 :: Int) >> (putListOf put . NLA.toList . NLA.flatten . H.extract $ x)
      S1DV x -> put (2 :: Int) >> put (V.toList x)
      S2DV x -> put (2 :: Int) >> put (V.toList x)
  get = do
    (nr :: Int) <- get
    case nr of
      1 -> do
       Just i <- fromStorable . V.fromList <$> getListOf get
       return i
      2 ->
        fromStorableV . V.fromList <$> get
      _ -> error "unexpected case in get in Serialize instance in Shape.hs"

-- Helper function for creating the number instances
n1 :: ( forall a. Floating a => a -> a ) -> S x -> S x
n1 f (S1D x)  = S1D (f x)
n1 f (S2D x)  = S2D (f x)
n1 f (S3D x)  = S3D (f x)
n1 f (S1DV x) = S1DV (f x)
n1 f (S2DV x) = S2DV (f x)


-- helper function for creating the number instances
n2 :: ( forall a. Floating a => a -> a -> a ) -> S x -> S x -> S x
n2 f (S1D x) (S1D y)    = S1D (f x y)
n2 f (S2D x) (S2D y)    = S2D (f x y)
n2 f (S3D x) (S3D y)    = S3D (f x y)
n2 f (S1DV x) (S1DV y)  = S1DV (f x y)
n2 f (S2DV x) (S2DV y)= S2DV (f x y)
n2 f x@S1DV{} (S1D y)   = n2 f x (S1DV $ extract y)
n2 f (S1D x) y@S1DV{}   = n2 f (S1DV $ extract x) y
n2 f x@(S2DV _) (S2D y) = n2 f x (S2DV (NLA.flatten . H.extract $ y))
n2 f (S2D x) y@(S2DV _) = n2 f (S2DV (NLA.flatten . H.extract $ x)) y

-- Helper function for creating the number instances
nk :: forall x. SingI x => RealNum -> S x
nk x = case (sing :: Sing x) of
  D1Sing SNat ->
    S1D (konst x)

  D2Sing SNat SNat ->
    S2D (konst x)

  D3Sing SNat SNat SNat ->
    S3D (konst x)
