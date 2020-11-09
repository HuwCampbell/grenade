{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE CPP                   #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE DeriveAnyClass        #-}
{-# LANGUAGE DeriveGeneric         #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE InstanceSigs          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE Strict                #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE UndecidableInstances  #-}
{-# OPTIONS_GHC -Wno-incomplete-uni-patterns #-}

module Grenade.Layers.FullyConnected (
    FullyConnected (..)
  , FullyConnected' (..)
  , randomFullyConnected
  , SpecFullyConnected (..)
  , specFullyConnected
  , fullyConnected
  ) where

import           Control.DeepSeq
import           Control.Monad.Primitive        (PrimBase, PrimState)
import           Data.Maybe                     (fromJust, fromMaybe)
import           Data.Reflection                (reifyNat)
import qualified Data.Vector.Storable           as V
import qualified Data.Vector.Storable           as U (unsafeFromForeignPtr0,
                                                      unsafeToForeignPtr0)
import           Foreign.ForeignPtr             (withForeignPtr)
import           GHC.Generics                   (Generic)
import           GHC.IO.Handle.Text             (memcpy)
import           GHC.TypeLits
import           System.Random.MWC              hiding (create)
#if MIN_VERSION_singletons(2,6,0)
import           Data.Singletons.TypeLits       (SNat (..))
#endif
import           Data.Proxy
import           Data.Serialize
import           Data.Singletons
import           Data.Singletons.Prelude.Num    ((%*))

import qualified Numeric.LinearAlgebra          as LA
import           Numeric.LinearAlgebra.Static

import           Control.Monad                  (void)
import           Data.Vector.Storable.Mutable   as VM
import           Foreign.Storable               (sizeOf)
import           System.IO.Unsafe               (unsafePerformIO)

import           Grenade.Core
import           Grenade.Dynamic
import           Grenade.Dynamic.Internal.Build
import           Grenade.Layers.Internal.CBLAS  as B
import           Grenade.Layers.Internal.Update
import           Grenade.Types
import           Grenade.Utils.LinearAlgebra
import           Grenade.Utils.ListStore

import           Debug.Trace


-- | A basic fully connected (or inner product) neural network layer.
data FullyConnected i o = FullyConnected
                        !(FullyConnected' i o)             -- Neuron weights
                        !(ListStore (FullyConnected' i o)) -- momentum store
                        !TempVectors                     -- Temporary vectors for fast implementations
                        deriving (Generic)

data TempVectors =
  TempVectors
    (V.Vector RealNum) -- ^ In vector size
    (V.Vector RealNum) -- ^ Out vector size
    (V.Vector RealNum) -- ^ Weight size
  deriving (NFData, Generic)

instance NFData (FullyConnected i o) where
  rnf (FullyConnected w store tmp) = rnf w `seq` rnf store `seq` rnf tmp

instance Show (FullyConnected i o) where
  show FullyConnected {} = "FullyConnected"


-- | How to store the data .
data FullyConnected' i o
  = FullyConnectedHMatrix
      !(R o)   -- ^ Bias
      !(L o i) -- ^ Activations
  | FullyConnectedHBLAS
    !(V.Vector RealNum) -- ^ Bias, Temporary vector of same size
    !(V.Vector RealNum) -- ^ Activations
  deriving (Generic)

instance Show (FullyConnected' i o) where
  show FullyConnectedHBLAS{}   = "FullyConnectedHBLAS"
  show FullyConnectedHMatrix{} = "FullyConnectedHMatrix"

instance NFData (FullyConnected' i o) where
  rnf (FullyConnectedHMatrix b w) = rnf b `seq` rnf w
  rnf (FullyConnectedHBLAS !b !w) = rnf b `seq` rnf w


instance (KnownNat i, KnownNat o, KnownNat (i * o)) => UpdateLayer (FullyConnected i o) where
  type Gradient (FullyConnected i o) = (FullyConnected' i o)
  type MomentumStore (FullyConnected i o) = ListStore (FullyConnected' i o)
  runUpdate opt@OptSGD {} x@(FullyConnected (FullyConnectedHMatrix oldBias oldActivations) store tmpVecs) (FullyConnectedHMatrix biasGradient activationGradient) =
    let (FullyConnectedHMatrix oldBiasMomentum oldMomentum) = getData opt x store
        VectorResultSGD newBias newBiasMomentum = descendVector opt (VectorValuesSGD oldBias biasGradient oldBiasMomentum)
        MatrixResultSGD newActivations newMomentum = descendMatrix opt (MatrixValuesSGD oldActivations activationGradient oldMomentum)
        newStore = setData opt x store (FullyConnectedHMatrix newBiasMomentum newMomentum)
     in FullyConnected (FullyConnectedHMatrix newBias newActivations) newStore tmpVecs
  runUpdate opt@OptAdam {} x@(FullyConnected (FullyConnectedHMatrix oldBias oldActivations) store tmpVecs) (FullyConnectedHMatrix biasGradient activationGradient) =
    let [FullyConnectedHMatrix oldMBias oldMActivations, FullyConnectedHMatrix oldVBias oldVActivations] = getData opt x store
        VectorResultAdam newBias newMBias newVBias = descendVector opt (VectorValuesAdam (getStep store) oldBias biasGradient oldMBias oldVBias)
        MatrixResultAdam newActivations newMActivations newVActivations =
          descendMatrix opt (MatrixValuesAdam (getStep store) oldActivations activationGradient oldMActivations oldVActivations)
        newStore = setData opt x store [FullyConnectedHMatrix newMBias newMActivations, FullyConnectedHMatrix newVBias newVActivations]
     in FullyConnected (FullyConnectedHMatrix newBias newActivations) newStore tmpVecs
  runUpdate opt@OptAdam {} x@(FullyConnected (FullyConnectedHBLAS oldBiasH oldActivationsH) store tmpVecs) (FullyConnectedHBLAS biasGradient activationGradient) =
      let oldBias = oldBiasH
          oldActivations = oldActivationsH
          (oldMBias, oldMActivations, oldVBias, oldVActivations) = case getData opt x store of -- In the first periods until the store is filled newData is called, which will generate FullyConnectedHMatrix instances!
            [FullyConnectedHBLAS oldMBias' oldMActivations', FullyConnectedHBLAS oldVBias' oldVActivations'] -> (oldMBias', oldMActivations', oldVBias', oldVActivations')
            [FullyConnectedHMatrix oldMBias' oldMActivations', FullyConnectedHMatrix oldVBias' oldVActivations'] -> (extract oldMBias', extractM oldMActivations', extract oldVBias', extractM oldVActivations')
            [FullyConnectedHBLAS oldMBias' oldMActivations', FullyConnectedHMatrix oldVBias' oldVActivations'] -> (oldMBias', oldMActivations', extract oldVBias', extractM oldVActivations')
            xs -> error $ "unexpected data in ListStore in FullyConnected HBLAS implementation: " ++ show xs
          VectorResultAdamV newBias newMBias newVBias = descendVectorV opt (VectorValuesAdamV (getStep store) oldBias biasGradient oldMBias oldVBias)
          MatrixResultAdamV newActivations newMActivations newVActivations =
            descendMatrixV opt (MatrixValuesAdamV (getStep store) oldActivations activationGradient oldMActivations oldVActivations)
          newStore = setData opt x store [FullyConnectedHBLAS newMBias newMActivations, FullyConnectedHBLAS newVBias newVActivations]
      in FullyConnected (FullyConnectedHBLAS newBias newActivations) newStore tmpVecs
    where extractM = V.concat . map extract . toColumns
  runUpdate OptSGD{} (FullyConnected FullyConnectedHMatrix{} _ _) _ = error "HBLAS is only implemented for OptAdam in runUpdate for FullyConnected layer"

instance (KnownNat i, KnownNat o, KnownNat (i * o)) => LayerOptimizerData (FullyConnected i o) (Optimizer 'SGD) where
  type MomentumDataType (FullyConnected i o) (Optimizer 'SGD) = FullyConnected' i o
  getData opt x store = head $ getListStore opt x store
  setData opt x store = setListStore opt x store . return
  newData _ _ = FullyConnectedHMatrix (konst 0) (konst 0)

instance (KnownNat i, KnownNat o, KnownNat (i * o)) => LayerOptimizerData (FullyConnected i o) (Optimizer 'Adam) where
  type MomentumDataType (FullyConnected i o) (Optimizer 'Adam) = FullyConnected' i o
  type MomentumExpOptResult (FullyConnected i o) (Optimizer 'Adam) = [FullyConnected' i o]
  getData = getListStore
  setData = setListStore
  newData _ _ = FullyConnectedHMatrix (konst 0) (konst 0)


instance (KnownNat i, KnownNat o) => FoldableGradient (FullyConnected' i o) where
  mapGradient f (FullyConnectedHMatrix bias activations) = FullyConnectedHMatrix (dvmap f bias) (dmmap f activations)
  mapGradient f (FullyConnectedHBLAS bias activations) = FullyConnectedHBLAS (V.map f bias) (V.map f activations)
  squaredSums (FullyConnectedHMatrix bias activations) = [sumV . squareV $ bias, sumM . squareM $ activations]
  squaredSums (FullyConnectedHBLAS bias activations) = [V.sum . V.map (^(2::Int)) $ bias, V.sum . V.map (^(2::Int)) $ activations]


-- -- | Computes mat * vec1 + vec2
-- matXVec :: S ('D2 i o) -> S ('D1 i) -> Maybe (S ('D1 o)) -> S ('D1 o)
-- matXVec (S2DV mat) (S1DV vec1) mVec2 = S1DV $ matXVec'  B.BlasNoTranspose mat B.BlasNoTranspose vec1 (unwrapS <$> mVec2)
--   where
--     unwrapS :: S ('D1 o) -> V.Vector RealNum
--     unwrapS (S1DV v2) = v2
--     unwrapS _ = error $ "unexpected third parameter type in matXVec: " ++ show mVec2

-- | Memory copy a vector from one to the other.
unsafeMemCopyVectorFromTo :: V.Vector RealNum -> V.Vector RealNum -> V.Vector RealNum
unsafeMemCopyVectorFromTo from to =
    unsafePerformIO $ do
      let (fromPtr, _) = U.unsafeToForeignPtr0 from
          (toPtr, _) = U.unsafeToForeignPtr0 to
      withForeignPtr fromPtr $ \fromPtr' ->
       withForeignPtr toPtr $ \toPtr' -> do
        void $ memcpy toPtr' fromPtr' (fromIntegral $ sizeOf (V.head from) * V.length to)
        return $ U.unsafeFromForeignPtr0 toPtr (V.length to)


-- | Computes vec2 = mat * vec1 + beta * vec2.
matXVec'' :: BlasTranspose -> V.Vector RealNum -> BlasTranspose -> V.Vector RealNum -> RealNum -> V.Vector RealNum -> V.Vector RealNum
matXVec'' trMat mat trVec1 vec1 beta vec2 =
  dgemmUnsafe BlasRowMajor trMat trVec1 (swapTranspose trMat (m, n)) (n, 1) (m, 1) 1.0 mat LDNormal vec1 LDNormal beta vec2 LDNormal `seq` vec2
  where n = V.length vec1
        m | V.length mat `mod` n /= 0 = error $ "matXVec input dimensions do not match: " ++ show (V.length mat) ++ " `mod` " ++ show n ++ " /= 0"
          | otherwise = V.length mat `div` n


-- | Computes mat * vec1 + vec2.
matXVec' :: BlasTranspose -> V.Vector RealNum -> BlasTranspose -> V.Vector RealNum -> Maybe (V.Vector RealNum) -> V.Vector RealNum
matXVec' trMat mat trVec1 vec1 mVec2 =
  dgemmUnsafe BlasRowMajor trMat trVec1 (swapTranspose trMat (m, n)) (n, 1) (m, 1) 1.0 mat LDNormal vec1 LDNormal 1.0 out LDNormal `seq` out
  where out = maybe (V.replicate m 0) gen mVec2
        gen vec2 = V.generate m (vec2 V.!)
        n = V.length vec1
        m | V.length mat `mod` n /= 0 = error $ "matXVec input dimensions do not match: " ++ show (V.length mat) ++ " `mod` " ++ show n ++ " /= 0"
          | otherwise = V.length mat `div` n

-- | Outer product of two vectors.
outer'' :: V.Vector RealNum -> V.Vector RealNum -> RealNum -> V.Vector RealNum -> V.Vector RealNum
outer'' vec1 vec2 beta mat =
  dgemmUnsafe BlasRowMajor B.BlasNoTranspose B.BlasNoTranspose (o, 1) (1, i) (o, i) 1.0 vec1 (LDCustom 1) vec2 (LDCustom i) beta mat (LDCustom i)
  where o = V.length vec1
        i = V.length vec2
        -- mat = maybe (V.replicate (i*o) 0) gen mMat
        -- gen mat' = V.generate (i*o) (mat' V.!)


-- | Outer product of two vectors.
outer' :: V.Vector RealNum -> V.Vector RealNum -> Maybe (V.Vector RealNum) -> V.Vector RealNum
outer' vec1 vec2 mMat =
  dgemmUnsafe BlasRowMajor B.BlasNoTranspose B.BlasNoTranspose (o, 1) (1, i) (o, i) 1.0 vec1 (LDCustom 1) vec2 (LDCustom i) 0.0 mat (LDCustom i)
  where o = V.length vec1
        i = V.length vec2
        mat = maybe (V.replicate (i*o) 0) gen mMat
        gen mat' = V.generate (i*o) (mat' V.!)


checkVecs :: V.Vector RealNum -> V.Vector RealNum -> Bool
checkVecs v1 v2 = V.length v1 == V.length v2 && and (zipWith (==) (toStr v1) (toStr v2))
  where
    toStr :: V.Vector RealNum -> [String]
    toStr v = map show $ map (round . (*10^5)) $ V.toList v


runForward :: forall i o . (KnownNat i, KnownNat o) => FullyConnected i o -> S (D1 i) -> (Tape (FullyConnected i o) ('D1 i) ('D1 o), S ('D1 o))
runForward (FullyConnected (FullyConnectedHMatrix wB wN) _ _) (S1D v) = (extract v, S1D (wB + wN #> v))
runForward (FullyConnected (FullyConnectedHMatrix wB wN) _ _) (S1DV v) = error "unexpected mixing of HBLAS and HMatrix in FullyConnected"
runForward lay@(FullyConnected FullyConnectedHBLAS{} _ _) (S1D v) = runForward lay (S1DV $ extract v)
runForward (FullyConnected (FullyConnectedHBLAS wB wN) _ (TempVectors _ wBTmp _)) (S1DV v) =

    let -- out' = matXVec'  B.BlasNoTranspose wN  B.BlasNoTranspose v (Just wB)
        -- wB' = V.generate (V.length wB) (wB V.!)

        !out' = matXVec'' B.BlasNoTranspose wN  B.BlasNoTranspose v 1.0 (unsafeMemCopyVectorFromTo wB wBTmp)
        -- check = extract ((vector (V.toList wB) :: R o) + (matrix (V.toList wN) :: L o i) #> (vector (V.toList v) :: R i))
    in -- (if checkVecs check out' then trace ("chk: " ++ show check) trace ("wN:  " ++ show wN) trace ("v:   " ++ show v) trace ("wB:  " ++ show wB) id else trace (show check) trace (show out) undefined )
      force (v, S1DV out')


runBackward :: forall i o . (KnownNat i, KnownNat o) => FullyConnected i o -> Tape (FullyConnected i o) ('D1 i) ('D1 o) -> S ('D1 o) -> (Gradient (FullyConnected i o), S ('D1 i))
runBackward (FullyConnected (FullyConnectedHMatrix _ wN) _ _) x (S1D dEdy) =
  let wB' = dEdy
      mm' = dEdy `outer` vector (V.toList x)
              -- calcluate derivatives for next step
      dWs = tr wN #> dEdy
   in (FullyConnectedHMatrix wB' mm', S1D dWs)
runBackward (FullyConnected (FullyConnectedHMatrix _ wN) _ _) x (S1DV dEdy) = error "unexpected mixing of HBLAS and HMatrix types in FullyConnected"
runBackward lay@(FullyConnected FullyConnectedHBLAS{} _ _) x (S1D dEdy) = runBackward lay x (S1DV $ extract dEdy)
runBackward (FullyConnected (FullyConnectedHBLAS _ wN) _ (TempVectors wIn _ wNTmp)) x (S1DV dEdy) =

  let
      mm' = outer'' dEdy x 0 wNTmp -- dgemmUnsafe BlasRowMajor B.BlasNoTranspose B.BlasNoTranspose (o, 1) (1, i) (o, i) 1.0 dEdy (LDCustom 1) x (LDCustom i) 0.0 mm (LDCustom i)
      -- mmCheck  = V.concat $ map (V.map realToFrac . extract) $ toRows $ ((vector (V.toList dEdy) :: R o) `outer` (vector (V.toList x) :: R i) :: L o i)
      -- mm = V.replicate (i*o) 0 :: V.Vector RealNum

    -- moments
      -- dWs' = matXVec' BlasTranspose wN BlasNoTranspose dEdy Nothing -- dgemmUnsafe BlasRowMajor B.BlasTranspose B.BlasNoTranspose (o, i) (o, 1) (i, 1) 1.0 wN LDNormal dEdy LDNormal 0.0 dWs LDNormal
      dWs' = matXVec'' BlasTranspose wN BlasNoTranspose dEdy 0 wIn -- dgemmUnsafe BlasRowMajor B.BlasTranspose B.BlasNoTranspose (o, i) (o, 1) (i, 1) 1.0 wN LDNormal dEdy LDNormal 0.0 dWs LDNormal

      -- dWs =  V.replicate i 0 :: V.Vector RealNum
      -- mmCheck2  = extract ((tr (matrix $ V.toList wN) :: L i o) #> (vector (V.toList dEdy) :: R o) :: R i) :: V.Vector RealNum
      --  check = extract $ ((matrix (V.toList wN) :: L o i) #> (vector (V.toList v) :: R i))
  in -- (if checkVecs mm' mmCheck then id else trace ("dEdy: " ++ show dEdy) trace ("x: " ++ show x) trace ("mm : " ++ show mm) trace ("dEdy R o: " ++ show (vector (V.toList dEdy) :: R o)) trace ("x R i: " ++ show (vector (V.toList x) :: R i) ) undefined)
     -- (if checkVecs dWs' mmCheck2 then id else trace ("wN': " ++ show wN) trace ("dEdy : " ++ show dEdy) trace ("dWs:   " ++ show dWs) trace ("\nL i o: " ++ show (tr (matrix $ V.toList wN) :: L i o)) trace ("R o: " ++ show (vector (V.toList dEdy) :: R o)) undefined)
     force (FullyConnectedHBLAS dEdy mm', S1DV dWs')

instance (KnownNat i, KnownNat o, KnownNat (i * o)) => Layer (FullyConnected i o) ('D1 i) ('D1 o) where
  type Tape (FullyConnected i o) ('D1 i) ('D1 o) = V.Vector RealNum
  runForwards = runForward   -- Do a matrix vector multiplication and return the result.
  runBackwards = runBackward -- Run a backpropogation step for a full connected layer.


instance (KnownNat i, KnownNat o) => Serialize (FullyConnected i o) where
  put (FullyConnected w ms tmp) = put w >> put ms >> put tmp
  get = FullyConnected <$> get <*> get <*> get

instance Serialize TempVectors where
  put (TempVectors inTmp outTmp wTmp) = put (V.toList inTmp) >> put (V.toList outTmp) >> put (V.toList wTmp)
  get = do
    inTmp <- V.fromList <$> get
    outTmp <- V.fromList <$> get
    wTmp <- V.fromList <$> get
    return $ TempVectors inTmp outTmp wTmp


instance (KnownNat i, KnownNat o) => Serialize (FullyConnected' i o) where
  put (FullyConnectedHMatrix b w) = do
    put (0 :: Int)
    putListOf put . LA.toList . extract $ b
    putListOf put . LA.toList . LA.flatten . extract $ w
  put (FullyConnectedHBLAS b w) = do
    put (1 :: Int)
    putListOf put . V.toList $ b
    putListOf put . V.toList $ w
  get = do
    (nr :: Int) <- get
    case nr of
      0 -> do
        let f  = fromIntegral $ natVal (Proxy :: Proxy i)
        b     <- maybe (fail "Vector of incorrect size") return . create . LA.fromList =<< getListOf get
        k     <- maybe (fail "Vector of incorrect size") return . create . LA.reshape f . LA.fromList =<< getListOf get
        return $ FullyConnectedHMatrix b k
      1 -> do
        b <- V.fromList <$> get
        w <- V.fromList <$> get
        return $ FullyConnectedHBLAS b w
      _ -> error $ "Unexpected nr in get in Serialize of FullyConnected' " ++ show nr


instance (KnownNat i, KnownNat o, KnownNat (i*o)) => RandomLayer (FullyConnected i o) where
  createRandomWith = randomFullyConnected


randomFullyConnected :: forall m i o . (PrimBase m, KnownNat i, KnownNat o, KnownNat (i*o))
                     => NetworkInitSettings -> Gen (PrimState m) -> m (FullyConnected i o)
randomFullyConnected (NetworkInitSettings m HMatrix) gen = do
  wN <- getRandomMatrix i o m gen
  wB <- getRandomVector i o m gen
  let wNTmp = V.empty :: V.Vector RealNum
      wBTmp = V.empty :: V.Vector RealNum
      wInTmp = V.empty :: V.Vector RealNum
  return $!! FullyConnected (FullyConnectedHMatrix wB wN) mkListStore (TempVectors wInTmp wBTmp wNTmp)
  where i = natVal (Proxy :: Proxy i)
        o = natVal (Proxy :: Proxy o)
randomFullyConnected (NetworkInitSettings m HBLAS) gen = do
  wB <- V.fromList <$> getRandomList i o o' m gen
  wN <- V.fromList <$> getRandomList i o (i' * o') m gen
  let wInTmp = V.replicate i' 0
      wBTmp = V.replicate o' 0
      wNTmp = V.replicate (i' * o') 0
  return $ FullyConnected (FullyConnectedHBLAS wB wN) mkListStore (TempVectors wInTmp wBTmp wNTmp)
  where
    i = natVal (Proxy :: Proxy i)
    i' = fromIntegral i
    o = natVal (Proxy :: Proxy o)
    o' = fromIntegral o

-------------------- DynamicNetwork instance --------------------

instance (KnownNat i, KnownNat o) => FromDynamicLayer (FullyConnected i o) where
  fromDynamicLayer _ _ _ = SpecNetLayer $ SpecFullyConnected (natVal (Proxy :: Proxy i)) (natVal (Proxy :: Proxy o))

instance ToDynamicLayer SpecFullyConnected where
  toDynamicLayer wInit gen (SpecFullyConnected nrI nrO) =
    reifyNat nrI $ \(pxInp :: (KnownNat i) => Proxy i) ->
      reifyNat nrO $ \(pxOut :: (KnownNat o') => Proxy o') ->
        case singByProxy pxInp %* singByProxy pxOut of
          SNat -> do
            (layer :: FullyConnected i o') <- randomFullyConnected wInit gen
            return $ SpecLayer layer (sing :: Sing ('D1 i)) (sing :: Sing ('D1 o'))

-- | Make a specification of a fully connected layer (see Grenade.Dynamic.Build for a user-interface to specifications).
specFullyConnected :: Integer -> Integer -> SpecNet
specFullyConnected nrI nrO = SpecNetLayer $ SpecFullyConnected nrI nrO


-- | A Fully-connected layer with input dimensions as given in last output layer and output dimensions specified. 1D only!
fullyConnected :: Integer -> BuildM ()
fullyConnected rows = do
  (inRows, _, _) <- buildRequireLastLayerOut Is1D
  buildAddSpec (SpecNetLayer $ SpecFullyConnected inRows rows)
  buildSetLastLayer (rows, 1, 1)


-------------------- GNum instances --------------------

instance (KnownNat i, KnownNat o) => GNum (FullyConnected i o) where
  s |* FullyConnected w store tmp = FullyConnected (s |* w) (s |* store) tmp
  FullyConnected w1 store1 tmp |+ FullyConnected w2 store2 _ = FullyConnected (w1 |+ w2) (store1 |+ store2) tmp
  gFromRational r = error "gFromRational is considered bad in FullyConnected!" -- FullyConnected (gFromRational r) mkListStore (gFromRational 0)

instance (KnownNat i, KnownNat o) => GNum (FullyConnected' i o) where
  s |* FullyConnectedHMatrix b w = FullyConnectedHMatrix (dvmap (fromRational s *) b) (dmmap (fromRational s *) w)
  FullyConnectedHMatrix b1 w1 |+ FullyConnectedHMatrix b2 w2 = FullyConnectedHMatrix (b1 + b2) (w1 + w2)
  gFromRational r = FullyConnectedHMatrix (fromRational r) (fromRational r)
