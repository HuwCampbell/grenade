{-# LANGUAGE BangPatterns #-}

module Grenade.Layers.Internal.CBLAS
    ( BlasOrder (..)
    , BlasTranspose (..)
    , dgemmUnsafe
    ) where

import qualified Data.Vector.Storable as V
import qualified Data.Vector.Storable as U (unsafeFromForeignPtr0, unsafeToForeignPtr0)
import           Foreign              (mallocForeignPtrArray, withForeignPtr)
import           Foreign.C.Types
import           Foreign.Ptr
import           System.IO.Unsafe     (unsafePerformIO)


-- | Newtype holding CINT for CblasRowMajor or CblasColMajor
newtype CBLAS_ORDERT =
  CBOInt CInt
  deriving (Eq, Show)

-- | Row or Colum major. CblasRowMajor CblasColMajor
data BlasOrder
  = BlasRowMajor
  | BlasColMajor
  deriving (Eq, Show)

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

-- | Used to make the tranpose CINT value
encodeTranspose :: BlasTranspose -> CBLAS_TRANSPOSET
encodeTranspose BlasNoTranspose     = CBLAS_TransposeT 111
encodeTranspose BlasTranspose       = CBLAS_TransposeT 112
encodeTranspose BlasConjTranspose   = CBLAS_TransposeT 113
encodeTranspose BlasConjNoTranspose = CBLAS_TransposeT 114

-- | Error text
mkDimText :: (Show a1, Show a2, Show a3, Show a4, Show a5, Show a6) => (a1, a2) -> (a3, a4) -> (a5, a6) -> String
mkDimText (ax, ay) (bx, by) (cx, cy) = "resulting dimensions: [" ++ show ax ++ "x" ++ show ay ++ "]*[" ++ show bx ++ "x" ++ show by ++ "]=[" ++ show cx ++ "x" ++ show cy ++ "]"


-- | Computes: C <- alpha*op( A )*op( B ) + beta*C, where op(X) may transpose the matrix X

--
-- dgemm, see http://www.netlib.org/lapack/explore-html/dc/d18/cblas__dgemm_8c_abae9e96e4ce9231245ea09a3f46bdba5.html#abae9e96e4ce9231245ea09a3f46bdba5 for the documentation.
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
dgemmUnsafe :: BlasOrder -- ^ Row or Column Major
            -> BlasTranspose -- ^ Transpose Matrix A
            -> BlasTranspose -- ^ Transpose Matrix B
            -> (Int, Int)    -- ^ Dimensions Matrix A
            -> (Int, Int)    -- ^ Dimensions Matrix B
            -> (Int, Int)    -- ^ Dimensions Matrix C
            -> Double
            -> V.Vector Double -- ^ A
            -> V.Vector Double -- ^ B
            -> Double          -- ^ Beta
            -> V.Vector Double -- ^ C
            -> V.Vector Double -- ^ Return new C
dgemmUnsafe order trA trB dimA dimB (cx, cy) alpha matrixA matrixB beta matrixC
  | isBadGemm = error $! "bad dimension args to GEMM: ax ay bx by cx cy: " ++ show [ax, ay, bx, by, cx, cy] ++ " \n\t" ++ mkDimText (ax, ay) (bx, by) (cx, cy)
  | otherwise =
    unsafePerformIO $ do
      let (aPtr, _) = U.unsafeToForeignPtr0 matrixA
          (bPtr, _) = U.unsafeToForeignPtr0 matrixB
          (cPtr, _) = U.unsafeToForeignPtr0 matrixC
      withForeignPtr aPtr $ \aPtr' ->
       withForeignPtr bPtr $ \bPtr' ->
        withForeignPtr cPtr $ \cPtr' -> do
         cblas_dgemm
           (encodeOrder order) -- order
           (encodeTranspose trA) -- transpose A
           (encodeTranspose trB) -- transpose B
           (fromIntegral ax) -- = cx
           (fromIntegral by) -- = cy
           (fromIntegral ay) -- = bx
           alpha
           aPtr'
           (fromIntegral ay)
           bPtr'
           (fromIntegral by)
           beta
           cPtr'
           (fromIntegral cy)
         return $ U.unsafeFromForeignPtr0 cPtr (cx * cy)
  where
    (ax, ay) = swap trA dimA
    (bx, by) = swap trB dimB
    swap BlasNoTranspose x        = x
    swap BlasTranspose (a, b)     = (b, a)
    swap BlasConjNoTranspose x    = x
    swap BlasConjTranspose (a, b) = (b, a)
    isBadGemm = minimum [ax, ay, bx, by, cx, cy] <= 0 || not (cx == ax && cy == by && ay == bx)


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
