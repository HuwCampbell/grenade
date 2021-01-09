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
    , memCopyVectorFromTo
    , unsafeMemZero
    , memZero
      -- more complicated, but direct, function calls
    , BlasTranspose (..)
    , swapTranspose
    , dgemmUnsafe
    , dgemvUnsafe
    , dgerUnsafe
    ) where

import           Control.Monad
import           Data.IORef
import           Data.Proxy
import qualified Data.Vector.Storable         as V
import           Foreign                      (withForeignPtr)
import           Foreign.C.Types
import           Foreign.Ptr
import           Foreign.Storable             (sizeOf)
import           GHC.IO.Handle.Text           (memcpy)
import           GHC.TypeLits
import qualified Numeric.LinearAlgebra        as LA
import qualified Numeric.LinearAlgebra.Static as LAS
import           System.IO.Unsafe             (unsafePerformIO)

import           Grenade.Layers.Internal.CUDA
import           Grenade.Types
import           Grenade.Utils.Vector

import           Debug.Trace

#define USE_DGEMM_ONLY 0

-- | Computes vec2 <- mat * vec1 + beta * vec2.
matXVec :: BlasTranspose -> V.Vector RealNum -> V.Vector RealNum -> RealNum -> V.Vector RealNum -> IO (V.Vector RealNum)
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


-- | Computes the outer product of two vectors: mat <- vec1 `outer` vec2
outerV :: V.Vector RealNum -> V.Vector RealNum -> IO (V.Vector RealNum)
outerV vec1 vec2 =
#if USE_DGEMM_ONLY
  dgemmUnsafe BlasNoTranspose BlasNoTranspose (o, 1) (1, i) 1.0 vec1 vec2 0 (createVectorUnsafe (i * o)) -- beta = 0 initialises the matrix
#else
  createVector (i * o) >>= memZero >>= dgerUnsafe (o, i) 1.0 vec1 vec2
#endif
  where o = V.length vec1
        i = V.length vec2


-- | Check two vectors if they are equal. For testing purposes.
checkVectors :: V.Vector RealNum -> V.Vector RealNum -> Bool
checkVectors v1 v2 = V.length v1 == V.length v2 && and (zipWith (==) (toStr v1) (toStr v2))
  where
    toStr :: V.Vector RealNum -> [String]
    toStr v = map (show . round . (*10^5)) $ V.toList v
{-# INLINE checkVectors #-}

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

encodeTransposeIntBool :: BlasTranspose -> Int
encodeTransposeIntBool BlasNoTranspose     = 0
encodeTransposeIntBool BlasTranspose       = 1
encodeTransposeIntBool BlasConjTranspose   = 1
encodeTransposeIntBool BlasConjNoTranspose = 0
{-# INLINE encodeTransposeIntBool #-}

swapTranspose :: BlasTranspose -> (Int, Int) -> (Int, Int)
swapTranspose BlasNoTranspose x        = x
swapTranspose BlasTranspose (a, b)     = (b, a)
swapTranspose BlasConjNoTranspose x    = x
swapTranspose BlasConjTranspose (a, b) = (b, a)
{-# INLINE swapTranspose #-}

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
            -> RealNum           -- ^ Alpha
            -> V.Vector RealNum  -- ^ A
            -> V.Vector RealNum  -- ^ B
            -> RealNum           -- ^ Beta
            -> V.Vector RealNum  -- ^ C
            -> IO (V.Vector RealNum)  -- ^ Return new C
dgemmUnsafe trA trB (axIn, ayIn) (bxIn, byIn) alpha matrixA matrixB beta matrixC
  | isBadGemm =
    error $!
    "bad dimension args to dgemmUnsafe: ax ay bx by cx cy: " ++
    show [ax, ay, bx, by, ax, by] ++ " matrix C length: " ++ show (V.length matrixC) ++ "\n\t" ++ mkDimText (ax, ay) (bx, by) (ax, by)
  | otherwise = do
      V.unsafeWith matrixA $ \aPtr' ->
        V.unsafeWith matrixB $ \bPtr' ->
          V.unsafeWith matrixC $ \cPtr' ->  do
#ifdef USE_FLOAT
            sgemm_direct
#else
            dgemm_direct
#endif
              (encodeTransposeIntBool trA) -- transpose A
              (encodeTransposeIntBool trB) -- transpose B
              (fromIntegral ax)     -- rows of C = rows of A transposed
              (fromIntegral by)     -- cols of C = cols of B transposed
              (fromIntegral ay)     -- k = cols of A transposed = rows of B transposed
              alpha
              aPtr'
              (fromIntegral axIn) -- LDA
              bPtr'
              (fromIntegral bxIn) -- LDB
              beta
              cPtr'
              (fromIntegral ax)   -- LDC
            return matrixC
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
            -> RealNum           -- ^ Alpha
            -> V.Vector RealNum  -- ^ A
            -> V.Vector RealNum  -- ^ X
            -> RealNum           -- ^ Beta
            -> V.Vector RealNum  -- ^ C
            -> IO (V.Vector RealNum)  -- ^ Return new C
dgemvUnsafe trA (m, k) alpha matrixA vecX beta vecY
  | ax /= V.length vecY || ay /= V.length vecX =
    error $!
    "bad dimension args to dgemvUnsafe: ax ay (length vecX) (length vecY): " ++
    show [ax, ay, V.length vecX, V.length vecY] ++ " \n\t" ++ mkDimText (ax, ay) (V.length vecX, 1) (m, 1)
  | otherwise = do
      V.unsafeWith matrixA $ \aPtr' ->
        V.unsafeWith vecX $ \xPtr' ->
          V.unsafeWith vecY $ \yPtr' -> do
#ifdef USE_FLOAT
            sgemv_direct
#else
            dgemv_direct
#endif
              (encodeTransposeIntBool trA)      -- transpose A
              (fromIntegral m)
              (fromIntegral k)
              alpha
              aPtr'
              (fromIntegral m)
              xPtr'
              1
              beta
              yPtr'
              1
            return vecY
  where
    (ax, ay) = swapTranspose trA (m, k)

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
dgerUnsafe :: (Int, Int)           -- ^ Dimensions of matrix A
           -> RealNum               -- ^ Alpha
           -> V.Vector RealNum      -- ^ X
           -> V.Vector RealNum      -- ^ C
           -> V.Vector RealNum      -- ^ A
           -> IO (V.Vector RealNum)  -- ^ Return new C
dgerUnsafe (ax, ay) alpha vecX vecY matrixA
  | ax /= len || ay /= V.length vecY =
    error $! "bad dimension args to dgerUnsafe: X Y ax ay: " ++ show [len, V.length vecY, ax, ay] ++ " \n\t" ++ mkDimText (ax, ay) (V.length vecX, 1) (V.length vecY, 1)
  | otherwise = do
      V.unsafeWith matrixA $ \aPtr' ->
        V.unsafeWith vecX $ \xPtr' ->
          V.unsafeWith vecY $ \yPtr' -> do
#ifdef USE_FLOAT
            sger_direct
#else
            dger_direct
#endif
              (fromIntegral ax)
              (fromIntegral ay)
              alpha
              xPtr'
              1
              yPtr'
              1
              aPtr'
              (fromIntegral ax)
            return matrixA
  where
    len = V.length vecX


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

foreign import ccall unsafe "dgemm_direct" dgemm_direct :: BLASGemmFunFFI Double Double
foreign import ccall unsafe "sgemm_direct" sgemm_direct :: BLASGemmFunFFI Float Float


-- |  Matrix mult for general dense matrices
type BLASGemvFunFFI scale el
  =  Int    -- transpose A: 1, not transpose A: 0
  -> CInt   -- m
  -> CInt   -- n
  -> scale  -- alpha
  -> Ptr el -- Matrix A
  -> CInt   -- LDA
  -> Ptr el
  -> CInt
  -> scale -- beta
  -> Ptr el
  -> CInt
  -> IO ()

foreign import ccall unsafe "dgemv_direct" dgemv_direct :: BLASGemvFunFFI Double Double
foreign import ccall unsafe "sgemv_direct" sgemv_direct :: BLASGemvFunFFI Float Float


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

foreign import ccall unsafe "dger_direct" dger_direct :: BlasGerxFunFFI Double Double
foreign import ccall unsafe "sger_direct" sger_direct :: BlasGerxFunFFI Float Float


-- toRows :: Int -> V.Vector Double -> [V.Vector Double]
-- toRows m vec = LA.toRows . reshapeF m . LA.vector . V.toList $ vec
--   where reshapeF r = LA.tr' . LA.reshape r

-- vec1 :: LAS.R 10
-- vec1 = LAS.vector [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.34868882831497655, 0.0, 1.4026932193043212e-2]

-- -- dEdy:
-- -- mmCheck : [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-4.8711473950841355e-2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-1.9595481318788644e-3]
-- -- mm' : [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-0.24602759633121374,-1.2277801012906878e-2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-9.89711206989971e-3,-4.939070836308897e-4,0.0,0.0]

-- vec3 :: V.Vector Double
-- vec3 = (
--   V.fromList [0.0, 0.0, 0.0, 0.0, (-0.13969898085418284)])

-- vec2 :: LAS.R 5
-- vec2 = LAS.vector [0.0, 0.0, 0.0, 0.0, -0.13969898085418284]

-- res = vec1 `LAS.outer` vec2

-- test =
--   toRows 10 $
--   outerV (LAS.extract vec1) (LAS.extract vec2) (V.replicate (LAS.size vec1 * LAS.size vec2) 10)
