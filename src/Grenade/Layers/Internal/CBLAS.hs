{-# LANGUAGE BangPatterns #-}

module Grenade.Layers.Internal.CBLAS
    ( BlasOrder (..)
    , BlasTranspose (..)
    , LeadingDimension (..)
    , dgemmUnsafe
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

-- | Error text
mkDimText :: (Show a1, Show a2, Show a3, Show a4, Show a5, Show a6) => (a1, a2) -> (a3, a4) -> (a5, a6) -> String
mkDimText (ax, ay) (bx, by) (cx, cy) = "resulting dimensions: [" ++ show ax ++ "x" ++ show ay ++ "]*[" ++ show bx ++ "x" ++ show by ++ "]=[" ++ show cx ++ "x" ++ show cy ++ "]"


data LeadingDimension
  = LDNormal -- ^ Column
  | LDSwap   -- ^ Row
  | LDCustom Int -- ^ Custom value
  deriving (Show, Eq, Ord)

fromLeadingDimension :: BlasOrder -> (Int, Int) -> LeadingDimension -> CInt
fromLeadingDimension BlasRowMajor (_,y) LDNormal  = fromIntegral y
fromLeadingDimension BlasRowMajor (x,_) LDSwap    = fromIntegral x
fromLeadingDimension BlasRowMajor _ (LDCustom nr) = fromIntegral nr
fromLeadingDimension BlasColMajor (x,y) LDNormal  = fromIntegral x
fromLeadingDimension BlasColMajor (_,y) LDSwap    = fromIntegral y
fromLeadingDimension BlasColMajor _ (LDCustom nr) = fromIntegral nr


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
dgemmUnsafe :: BlasOrder        -- ^ Row or Column Major
            -> BlasTranspose    -- ^ Transpose Matrix A
            -> BlasTranspose    -- ^ Transpose Matrix B
            -> (Int, Int)       -- ^ Dimensions Matrix A
            -> (Int, Int)       -- ^ Dimensions Matrix B
            -> (Int, Int)       -- ^ Dimensions Matrix C
            -> Double           -- ^ Alpha
            -> V.Vector Double  -- ^ A
            -> LeadingDimension -- ^ LDA, Leading dimension of A
            -> V.Vector Double  -- ^ B
            -> LeadingDimension -- ^ LDB, Leading dimension of B
            -> Double           -- ^ Beta
            -> V.Vector Double  -- ^ C
            -> LeadingDimension -- ^ LDC, Leading dimension of C
            -> V.Vector Double  -- ^ Return new C
dgemmUnsafe order trA trB dimA dimB (cx, cy) alpha matrixA ldA matrixB ldB beta matrixC ldC
  | isBadGemm = error $! "bad dimension args to GEMM: ax ay bx by cx cy: " ++ show [ax, ay, bx, by, cx, cy] ++ " \n\t" ++ mkDimText (ax, ay) (bx, by) (cx, cy)
  | otherwise =
    unsafePerformIO $ do
      let (aPtr, _) = U.unsafeToForeignPtr0 matrixA
          (bPtr, _) = U.unsafeToForeignPtr0 matrixB
          (cPtr, _) = U.unsafeToForeignPtr0 matrixC
      withForeignPtr aPtr $ \aPtr' ->
       withForeignPtr bPtr $ \bPtr' ->
        withForeignPtr cPtr $ \cPtr' -> do
         -- putStrLn $ "call: cblas_dgemm " ++ unwords [ show order
         --                                            , show trA -- transpose A
         --                                            , show trB -- transpose B
         --                                            , show ax -- = ax = cx if NoTranspose
         --                                            , show by -- = cy
         --                                            , show ay -- = bx
         --                                            , show alpha
         --                                            , "A"
         --                                            , show (fromLeadingDimension order (ax ,ay) ldA)
         --                                            , "B"
         --                                            , show (fromLeadingDimension order (bx, by) ldB)
         --                                            , show beta
         --                                            , "C"
         --                                            , show (fromLeadingDimension order (cx, cy) ldC)]

         cblas_dgemm
           (encodeOrder order) -- order
           (encodeTranspose trA) -- transpose A
           (encodeTranspose trB) -- transpose B
           (fromIntegral ax) -- = ax = cx
           (fromIntegral by) -- = cy = by
           (fromIntegral ay) -- = bx = ay
           alpha
           aPtr'
           (fromLeadingDimension order dimA ldA)
           bPtr'
           (fromLeadingDimension order dimB ldB)
           beta
           cPtr'
           (fromLeadingDimension order (cx,cy) ldC)
         return $ U.unsafeFromForeignPtr0 cPtr (cx * cy)
  where
    (ax, ay) = swapTranspose trA dimA
    (bx, by) = swapTranspose trB dimB
    isBadGemm = minimum [ax, ay, bx, by, cx, cy] <= 0 || not (cx == ax && cy == by && ay == bx)

swapTranspose :: BlasTranspose -> (Int, Int) -> (Int, Int)
swapTranspose BlasNoTranspose x        = x
swapTranspose BlasTranspose (a, b)     = (b, a)
swapTranspose BlasConjNoTranspose x    = x
swapTranspose BlasConjTranspose (a, b) = (b, a)


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
