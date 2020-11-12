{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE CPP                 #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeOperators       #-}

module Grenade.Layers.Internal.BLAS
    ( -- easy interface
      matXVec
    , outerV
    , checkVectors
    , unsafeMemCopyVectorFromTo
    , unsafeMemZero
      -- conversions according to column major type
    , toLayerShape
    , toS1D
    , fromS1D
    , toS2D
    , fromS2D
    , toRowMajorVector
    , fromRowMajorVector
      -- more complicated, but direct, function calls
    , BlasTranspose (..)
    , swapTranspose
    , dgemmUnsafe
    , dgemvUnsafe
    , dgerUnsafe
    ) where

import           Control.Monad
import           Data.Maybe                   (fromMaybe)
import qualified Data.Vector.Storable         as V
import qualified Data.Vector.Storable         as U (unsafeFromForeignPtr0,
                                                    unsafeToForeignPtr0)
import qualified Data.Vector.Storable.Mutable as VM
import           Foreign                      (withForeignPtr)
import           Foreign.C.Types
import           Foreign.Ptr
import           Foreign.Storable             (sizeOf)
import           GHC.IO.Handle.Text           (memcpy)
import           GHC.TypeLits
import qualified Numeric.LinearAlgebra.Static as LAS
import           System.IO.Unsafe             (unsafePerformIO)

import           Grenade.Core.Shape
import           Grenade.Types

import           Debug.Trace


-- DGEMM seems to be a little faster, and CBLAS has some overhead. So we aim for implementing all operations with DGEMM and use the direct calls to BLAS

#define USE_DGEMM_ONLY 1
#define USE_CBLAS 0


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


-- | Write zero to all elements in a vector.
unsafeMemZero :: V.Vector RealNum -> V.Vector RealNum
unsafeMemZero vec =
  unsafePerformIO $ do
    let (vecPtr, _) = U.unsafeToForeignPtr0 vec
    withForeignPtr vecPtr $ \vecPtr' -> void $ memset vecPtr' 0 (fromIntegral $ sizeOf (0 :: RealNum) * V.length vec)
    return $ U.unsafeFromForeignPtr0 vecPtr (V.length vec)

foreign import ccall unsafe "string.h" memset  :: Ptr a -> CInt  -> CSize -> IO (Ptr a)


-- | Computes vec2 <- mat * vec1 + beta * vec2.
matXVec :: BlasTranspose -> V.Vector RealNum -> V.Vector RealNum -> RealNum -> V.Vector RealNum -> V.Vector RealNum
matXVec trMat mat vec1 beta vec2 =
#if USE_DGEMM_ONLY
  dgemmUnsafe trMat BlasNoTranspose (m, k) (ay, 1) 1.0 mat vec1 beta vec2
# else
  dgemvUnsafe trMat (m, k) 1.0 mat vec1 beta vec2
#endif
  where
    ay = V.length vec1
    ax = V.length vec2
    (m, k) = swapTranspose trMat (ax, ay)
{-# INLINE matXVec #-}


-- | Computes the outer product of two vectors: mat <- vec1 `outer` vec2
outerV :: V.Vector RealNum -> V.Vector RealNum -> V.Vector RealNum -> V.Vector RealNum
outerV vec1 vec2 mat =
#if USE_DGEMM_ONLY
  dgemmUnsafe BlasNoTranspose BlasNoTranspose (o, 1) (1, i) 1.0 vec1 vec2 0 mat -- beta = 0 initialises the matrix
#else
  dgerUnsafe (o, i) 1.0 vec1 vec2 (unsafeMemZero mat)
#endif
  where o = V.length vec1
        i = V.length vec2
{-# INLINE outerV #-}


-- | Check two vectors if they are equal. For testing purposes.
checkVectors :: V.Vector RealNum -> V.Vector RealNum -> Bool
checkVectors v1 v2 = V.length v1 == V.length v2 && and (zipWith (==) (toStr v1) (toStr v2))
  where
    toStr :: V.Vector RealNum -> [String]
    toStr v = map (show . round . (*10^5)) $ V.toList v


-- | Newtype holding CINT for CblasRowMajor or CblasColMajor
newtype CBLAS_ORDERT =
  CBOInt CInt
  deriving (Eq, Show)


-- | Row or Colum major. CblasRowMajor CblasColMajor
data BlasOrder
  = BlasRowMajor
  | BlasColMajor
  deriving (Eq, Show)


-- | Column major seems to be a little faster (which makes sense as BLAS is in column major mode and CBLAS does not need change anything).
order :: BlasOrder
#if USE_CBLAS
order = BlasColMajor -- BlasRowMajor
#else
order = BlasColMajor -- no choice, must use ColMajor!
#endif


-- | Used to generate the Order (Row/Colummajor) CINT value.
encodeOrder :: BlasOrder -> CBLAS_ORDERT
encodeOrder BlasRowMajor = CBOInt 101
encodeOrder BlasColMajor = CBOInt 102

-- | Converts the given vector to the correct layer shape.
toLayerShape :: S i -> S x -> S x
toLayerShape x y = case (x, y) of
  (S1D{}, S1DV{}) -> toS1D y
  (S1DV{}, S1D{}) -> fromS1D y
  (S2D{}, S2DV{}) -> toS2D y
  (S2DV{}, S2D{}) -> fromS2D y
  -- (S3D{}, S2DV{}) -> y
  -- (S3D{}, S1DV{}) -> y
  _               -> y

toS1D :: S ('D1 l) -> S ('D1 l)
toS1D (S1DV vec) = S1D (fromMaybe (error $ "wrong length of vector with " ++ show (V.length vec) ++ " in toS1D ") $ LAS.create vec)
toS1D x@S1D{} = x

fromS1D :: S ('D1 l) -> S ('D1 l)
fromS1D (S1D vec) = S1DV (LAS.extract vec)
fromS1D x@S1DV{}  = x

-- | Convert from vector representation.
toS2D :: S ('D2 i j) -> S ('D2 i j)
toS2D (S2DV vec) =
  case order of
    BlasColMajor -> S2D $ LAS.tr $ LAS.matrix (V.toList vec)
    BlasRowMajor -> S2D $ LAS.matrix (V.toList vec)
toS2D x@S2D{}    = x

-- | Convert to vector representation.
fromS2D :: S ('D2 i j) -> S ('D2 i j)
fromS2D (S2D mat) =
  case order of
    BlasColMajor -> S2DV $ V.concat $ map LAS.extract $ LAS.toColumns mat
    BlasRowMajor -> S2DV $ V.concat $ map LAS.extract $ LAS.toRows mat
fromS2D x@S2DV{} = x

toRowMajorVector :: S ('D2 i j) -> V.Vector RealNum
toRowMajorVector x = case fromS2D x of
  S2DV vec -> vec
  _        -> error "unexpected return from fromS2D in toRowMajorVector"

fromRowMajorVector :: forall i j . (KnownNat i, KnownNat j) => V.Vector RealNum -> S ('D2 i j)
fromRowMajorVector vec =
  case order of
    BlasColMajor -> S2D $ LAS.tr $ LAS.matrix (V.toList vec)
    BlasRowMajor -> S2D $ LAS.matrix (V.toList vec)


-- | Newtype holding CINT for Transpose values.
newtype CBLAS_TRANSPOSET =
  CBLAS_TransposeT CInt
  deriving (Eq, Show)

-- | Transpose values
data BlasTranspose
  = BlasNoTranspose
  | BlasTranspose
  | BlasConjTranspose
  | BlasConjNoTranspose
  deriving (Eq, Show)

-- | Used to make the tranpose CINT value
encodeTranspose :: BlasTranspose -> CBLAS_TRANSPOSET
encodeTranspose BlasNoTranspose     = CBLAS_TransposeT 111
encodeTranspose BlasTranspose       = CBLAS_TransposeT 112
encodeTranspose BlasConjTranspose   = CBLAS_TransposeT 113
encodeTranspose BlasConjNoTranspose = CBLAS_TransposeT 114

-- | Used to make the tranpose CINT value
encodeTransposeChar :: BlasTranspose -> CChar
encodeTransposeChar BlasNoTranspose     = CChar 78 -- 'N'
encodeTransposeChar BlasTranspose       = CChar 84 -- 'T'
encodeTransposeChar BlasConjTranspose   = CChar 84 -- 'T'
encodeTransposeChar BlasConjNoTranspose = CChar 78 -- 'N'

encodeTransposeIntBool :: BlasTranspose -> Int
encodeTransposeIntBool BlasNoTranspose     = 0
encodeTransposeIntBool BlasTranspose       = 1
encodeTransposeIntBool BlasConjTranspose   = 1
encodeTransposeIntBool BlasConjNoTranspose = 0


swapTranspose :: BlasTranspose -> (Int, Int) -> (Int, Int)
swapTranspose BlasNoTranspose x        = x
swapTranspose BlasTranspose (a, b)     = (b, a)
swapTranspose BlasConjNoTranspose x    = x
swapTranspose BlasConjTranspose (a, b) = (b, a)

-- | Error text
mkDimText :: (Show a1, Show a2, Show a3, Show a4, Show a5, Show a6) => (a1, a2) -> (a3, a4) -> (a5, a6) -> String
mkDimText (ax, ay) (bx, by) (cx, cy) = "resulting dimensions: [" ++ show ax ++ "x" ++ show ay ++ "]*[" ++ show bx ++ "x" ++ show by ++ "]=[" ++ show cx ++ "x" ++ show cy ++ "]"


-- | Computes: C <- alpha*op( A )*op( B ) + beta*C, where op(X) may transpose the matrix X
--
-- dgemm, see http://www.netlib.org/lapack/explore-html/d1/d54/group__double__blas__level3_gaeda3cbd99c8fb834a60a6412878226e1.html for the documentation.
--
-- void cblas_dgemm (
--              const CBLAS_LAYOUT      layout,
--              const CBLAS_TRANSPOSE   TransA,
--              const CBLAS_TRANSPOSE   TransB,
--              const int       M,
--              const int       N,
--              const int       K,
--              const double    alpha,
--              const double *          A,
--              const int       lda,
--              const double *          B,
--              const int       ldb,
--              const double    beta,
--              double *        C,
--              const int       ldc
-- 	)
{-# NOINLINE dgemmUnsafe #-}
dgemmUnsafe :: BlasTranspose    -- ^ Transpose Matrix A
            -> BlasTranspose    -- ^ Transpose Matrix B
            -> (Int, Int)       -- ^ Rows and cols of A on entry (not transposed)
            -> (Int, Int)       -- ^ Rows and Cols of B on entry (not transposed)
            -> Double           -- ^ Alpha
            -> V.Vector Double  -- ^ A
            -> V.Vector Double  -- ^ B
            -> Double           -- ^ Beta
            -> V.Vector Double  -- ^ C
            -> V.Vector Double  -- ^ Return new C
dgemmUnsafe trA trB (axIn, ayIn) (bxIn, byIn) alpha matrixA matrixB beta matrixC
  | isBadGemm =
    error $!
    "bad dimension args to dgemmUnsafe: ax ay bx by cx cy: " ++
    show [ax, ay, bx, by, ax, by] ++ " matrix C length: " ++ show (V.length matrixC) ++ "\n\t" ++ mkDimText (ax, ay) (bx, by) (ax, by)
  | otherwise =
    unsafePerformIO $ do
      let (aPtr, _) = U.unsafeToForeignPtr0 matrixA
          (bPtr, _) = U.unsafeToForeignPtr0 matrixB
          (cPtr, _) = U.unsafeToForeignPtr0 matrixC
      withForeignPtr aPtr $ \aPtr' ->
        withForeignPtr bPtr $ \bPtr' ->
          withForeignPtr cPtr $ \cPtr' -> do
#if USE_CBLAS
            cblas_dgemm
              (encodeOrder order)   -- order
              (encodeTranspose trA) -- transpose A
              (encodeTranspose trB) -- transpose B
              (fromIntegral ax)     -- rows of C = rows of A transposed
              (fromIntegral by)     -- cols of C = cols of B transposed
              (fromIntegral ay)     -- k = cols of A transposed = rows of B transposed
              alpha
              aPtr'
              (ldOrder order (axIn, ayIn)) -- LDA
              bPtr'
              (ldOrder order (bxIn, byIn)) -- LDB
              beta
              cPtr'
              (ldOrder order (ax, by))   -- LDC
#else
            dgemm_direct
              (encodeTransposeIntBool trA) -- transpose A
              (encodeTransposeIntBool trB) -- transpose B
              (fromIntegral ax)     -- rows of C = rows of A transposed
              (fromIntegral by)     -- cols of C = cols of B transposed
              (fromIntegral ay)     -- k = cols of A transposed = rows of B transposed
              alpha
              aPtr'
              (ldOrder order (axIn, ayIn)) -- LDA
              bPtr'
              (ldOrder order (bxIn, byIn)) -- LDB
              beta
              cPtr'
              (ldOrder order (ax, by))   -- LDC
#endif
            return $ U.unsafeFromForeignPtr0 cPtr (ax * by)
  where
    (ax, ay) = swapTranspose trA (axIn, ayIn)
    (bx, by) = swapTranspose trB (bxIn, byIn)
    isBadGemm = minimum [ax, ay, bx, by] <= 0 || not (ax * by == V.length matrixC && ay == bx)


-- | Computes: Y <- alpha*op( A )*X + beta*Y, where op(A) may transpose the matrix A
--
-- dgemv, see http://www.netlib.org/lapack/explore-html/d7/d15/group__double__blas__level2_gadd421a107a488d524859b4a64c1901a9.html#gadd421a107a488d524859b4a64c1901a9
--
-- void cblas_dgemv 	(
--              const CBLAS_LAYOUT  	layout,
-- 		const CBLAS_TRANSPOSE  	TransA,
-- 		const int  	M,
-- 		const int  	N,
-- 		const double  	alpha,
-- 		const double *  	A,
-- 		const int  	lda,
-- 		const double *  	X,
-- 		const int  	incX,
-- 		const double  	beta,
-- 		double *  	Y,
-- 		const int  	incY
-- 	)
{-# NOINLINE dgemvUnsafe #-}
dgemvUnsafe :: BlasTranspose    -- ^ Transpose Matrix
            -> (Int, Int)       -- ^ rows and cols of A on entry (not transposed)
            -> Double           -- ^ Alpha
            -> V.Vector Double  -- ^ A
            -> V.Vector Double  -- ^ X
            -> Double           -- ^ Beta
            -> V.Vector Double  -- ^ C
            -> V.Vector Double  -- ^ Return new C
dgemvUnsafe trA (m, k) alpha matrixA vecX beta vecY
  | ax /= V.length vecY || ay /= V.length vecX =
    error $!
    "bad dimension args to dgemvUnsafe: ax ay (length vecX) (length vecY): " ++
    show [ax, ay, V.length vecX, V.length vecY] ++ " \n\t" ++ mkDimText (ax, ay) (V.length vecX, 1) (m, 1)
  | otherwise =
    unsafePerformIO $ do
      let (aPtr, _) = U.unsafeToForeignPtr0 matrixA
          (xPtr, _) = U.unsafeToForeignPtr0 vecX
          (yPtr, _) = U.unsafeToForeignPtr0 vecY
      withForeignPtr aPtr $ \aPtr' ->
        withForeignPtr xPtr $ \xPtr' ->
          withForeignPtr yPtr $ \yPtr' -> do
#if USE_CBLAS
            cblas_dgemv
              (encodeOrder order) -- order is always column major!
              (encodeTranspose trA)      -- transpose A
              (fromIntegral m)
              (fromIntegral k)
              alpha
              aPtr'
              (ldOrder order (m, k))
              xPtr'
              1
              beta
              yPtr'
              1
#else
            dgemv_direct
              (encodeTransposeIntBool trA)      -- transpose A
              (fromIntegral m)
              (fromIntegral k)
              alpha
              aPtr'
              (ldOrder order (m, k))
              xPtr'
              1
              beta
              yPtr'
              1
            return $ U.unsafeFromForeignPtr0 yPtr ax
#endif
  where
    (ax, ay) = swapTranspose trA (m, k)

ldOrder :: BlasOrder -> (Int, Int) -> CInt
ldOrder BlasColMajor = fromIntegral . fst
ldOrder BlasRowMajor = fromIntegral . snd


-- | Computes: A <- alpha*X*Y^T + A
--
-- dger, see http://www.netlib.org/lapack/explore-html/d7/d15/group__double__blas__level2_ga458222e01b4d348e9b52b9343d52f828.html#ga458222e01b4d348e9b52b9343d52f828
--
-- void cblas_dger 	(
--              const CBLAS_LAYOUT  	layout,
-- 		const int  	M,
-- 		const int  	N,
-- 		const double  	alpha,
-- 		const double *  	X,
-- 		const int  	incX,
-- 		const double *  	Y,
-- 		const int  	incY,
-- 		double *  	A,
-- 		const int  	lda
-- 	)
{-# NOINLINE dgerUnsafe #-}
dgerUnsafe :: (Int, Int)       -- ^ Dimensions of matrix A
           -> Double           -- ^ Alpha
           -> V.Vector Double  -- ^ X
           -> V.Vector Double  -- ^ C
           -> V.Vector Double  -- ^ A
           -> V.Vector Double  -- ^ Return new C
dgerUnsafe (ax, ay) alpha vecX vecY matrixA
  | ax /= len || ay /= V.length vecY =
    error $! "bad dimension args to dgerUnsafe: X Y ax ay: " ++ show [len, V.length vecY, ax, ay] ++ " \n\t" ++ mkDimText (ax, ay) (V.length vecX, 1) (V.length vecY, 1)
  | otherwise =
    unsafePerformIO $ do
      let (aPtr, _) = U.unsafeToForeignPtr0 matrixA
          (xPtr, _) = U.unsafeToForeignPtr0 vecX
          (yPtr, _) = U.unsafeToForeignPtr0 vecY
      withForeignPtr aPtr $ \aPtr' ->
        withForeignPtr xPtr $ \xPtr' ->
          withForeignPtr yPtr $ \yPtr' -> do
#if USE_CBLAS
            cblas_dger
              (encodeOrder order)
              (fromIntegral ax)
              (fromIntegral ay)
              alpha
              xPtr'
              1
              yPtr'
              1
              aPtr'
              (ldOrder order (ax, ay))
#else
            dger_direct
              (fromIntegral ax)
              (fromIntegral ay)
              alpha
              xPtr'
              1
              yPtr'
              1
              aPtr'
              (ldOrder order (ax, ay))
#endif
            return $ U.unsafeFromForeignPtr0 aPtr (V.length matrixA)
  where
    len = V.length vecX


-- |  Matrix mult for general dense matrices
type GemmFunFFI scale el
  = CBLAS_ORDERT
  -> CBLAS_TRANSPOSET
  -> CBLAS_TRANSPOSET
  -> CInt
  -> CInt
  -> CInt
  -> {- scal A * B -} scale
  -> {- Matrix A-} Ptr el
  -> CInt -> {- B -} Ptr el
  -> CInt
  -> scale
  -> {- C -}  Ptr el
  -> CInt
  -> IO ()

foreign import ccall unsafe "cblas_dgemm"
    cblas_dgemm :: GemmFunFFI Double Double

-- |  Matrix mult for general dense matrices
type BLASGemmFunFFI scale el
  =  Int -- transpose A: 1, not transpose A: 0
  -> Int -- transpose B: 1, not transpose B: 0
  -> CInt -- m
  -> CInt -- n
  -> CInt -- k
  -> {- scal A * B -} scale  -- alpha
  -> {- Matrix A-} Ptr el    -- A
  -> CInt                    -- LDA
  -> {- B -} Ptr el
  -> CInt
  -> scale                   -- beta
  -> {- C -}  Ptr el
  -> CInt
  -> IO ()

foreign import ccall unsafe "dgemm_direct"
    dgemm_direct :: BLASGemmFunFFI Double Double


              -- (encodeTransposeChar trA) -- transpose A
              -- (encodeTransposeChar trB) -- transpose B
              -- (fromIntegral ax)     -- rows of C = rows of A transposed
              -- (fromIntegral by)     -- cols of C = cols of B transposed
              -- (fromIntegral ay)     -- k = cols of A transposed = rows of B transposed
              -- alpha
              -- aPtr'
              -- (ldOrder order (axIn, ayIn)) -- LDA
              -- bPtr'
              -- (ldOrder order (bxIn, byIn)) -- LDB
              -- beta
              -- cPtr'
              -- (ldOrder order (ax, by))   -- LDC


type GemvFunFFI sc el
  =  CBLAS_ORDERT
  -> CBLAS_TRANSPOSET
  -> CInt
  -> CInt
  -> sc
  -> Ptr el
  -> CInt
  -> Ptr el
  -> CInt
  -> sc
  -> Ptr el
  -> CInt
  -> IO ()

foreign import ccall unsafe "cblas_dgemv"
    cblas_dgemv :: GemvFunFFI Double Double


-- |  Matrix mult for general dense matrices
type BLASGemvFunFFI scale el
  =  Int -- transpose A: 1, not transpose A: 0
  -> CInt -- m
  -> CInt -- n
  -> {- scal A * B -} scale  -- alpha
  -> {- Matrix A-} Ptr el    -- A
  -> CInt                    -- LDA
  -> {- B -} Ptr el
  -> CInt
  -> scale                   -- beta
  -> {- C -}  Ptr el
  -> CInt
  -> IO ()

foreign import ccall unsafe "dgemv_direct"
    dgemv_direct :: BLASGemvFunFFI Double Double


type GerxFunFFI scale el
  =  CBLAS_ORDERT
  -> CInt
  -> CInt
  -> scale
  -> Ptr el
  -> CInt
  -> Ptr el
  -> CInt
  -> Ptr el
  -> CInt
  -> IO ()

foreign import ccall unsafe "cblas_dger" cblas_dger ::
        GerxFunFFI Double Double


type BlasGerxFunFFI scale el
  =  CInt
  -> CInt
  -> scale
  -> Ptr el
  -> CInt
  -> Ptr el
  -> CInt
  -> Ptr el
  -> CInt
  -> IO ()

foreign import ccall unsafe "dger_direct" dger_direct ::
        BlasGerxFunFFI Double Double
