{-# LANGUAGE BangPatterns #-}

module Grenade.Layers.Internal.CBLAS
    ( BlasTranspose (..)
    , dgemmUnsafe
    , dgemvUnsafe
    , dgerUnsafe
    , swapTranspose
    ) where

import qualified Data.Vector.Storable as V
import qualified Data.Vector.Storable as U (unsafeFromForeignPtr0, unsafeToForeignPtr0)
import           Foreign              (mallocForeignPtrArray, withForeignPtr)
import           Foreign.C.Types
import           Foreign.Ptr
import           System.IO.Unsafe     (unsafePerformIO)

import           Debug.Trace


-- | Newtype holding CINT for CblasRowMajor or CblasColMajor
newtype CBLAS_ORDERT =
  CBOInt CInt
  deriving (Eq, Show)

-- | Row or Colum major. CblasRowMajor CblasColMajor
data BlasOrder
  = BlasRowMajor
  | BlasColMajor
  deriving (Eq, Show)

-- | Column major seems to be a little faster.
order :: BlasOrder
order = BlasColMajor -- BlasRowMajor


-- | Used to generate the Order (Row/Colummajor) CINT value.
encodeOrder :: BlasOrder -> CBLAS_ORDERT
encodeOrder BlasRowMajor = CBOInt 101
encodeOrder BlasColMajor = CBOInt 102

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
            return $ U.unsafeFromForeignPtr0 yPtr ax
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
  =  CBLAS_TRANSPOSET
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

foreign import ccall unsafe "dgemm"
    dgemm :: BLASGemmFunFFI Double Double


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
