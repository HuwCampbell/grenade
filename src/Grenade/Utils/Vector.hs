{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE GADTs            #-}
{-# LANGUAGE RankNTypes       #-}
module Grenade.Utils.Vector
    ( parMapVector
    , parZipWithVector
    ) where


import           Control.Monad.ST
import           Control.Parallel.Strategies
import qualified Data.Vector.Storable         as V
import qualified Data.Vector.Storable.Mutable as VM
import           GHC.Conc                     (numCapabilities)
import           Grenade.Types

parMapVector :: (RealNum -> RealNum) -> V.Vector RealNum -> V.Vector RealNum
parMapVector f vec = V.modify sliced vec
  where
    maxSize = max 10 $ V.length vec `div` (numCapabilities `div` 3) -- a third seems to be a good value. Usually we have 2 threads per CPU and we aim for a little less than that value
    sliced :: forall s. V.MVector s RealNum -> ST s ()
    sliced vec'
      | len <= maxSize = mapM_ (VM.modify vec' f) [0 .. VM.length vec' - 1]
      | otherwise = do
        sliced v1 `using` rpar
        sliced v2 `using` rpar
      where
        (v1, v2) = VM.splitAt idx vec' -- this uses unsafeSlice, e.g. does not create a new vector
        idx = len `div` 2
        len = VM.length vec'


parZipWithVector :: (RealNum -> RealNum -> RealNum) -> V.Vector RealNum -> V.Vector RealNum -> V.Vector RealNum
parZipWithVector f vec1 vec2 = V.modify sliced shorter
  where
    (shorter, longer) | V.length vec1 <= V.length vec2 = (vec1, vec2)
            | otherwise = (vec2, vec1)
    maxSize = max 10 $ V.length shorter `div` numCapabilities
    sliced :: forall s. V.MVector s RealNum -> ST s ()
    sliced v1 = sliced' v1 longer
    sliced' :: forall s. V.MVector s RealNum -> V.Vector RealNum -> ST s ()
    sliced' v1 v2
       | len <= maxSize = mapM_ (\idx -> VM.modify v1 (\val1 -> f val1 (v2 V.! idx)) idx) [0 .. VM.length v1 - 1]
       | otherwise = do
         sliced' v1First v2First `using` rpar
         sliced' v1Second v2Second `using` rpar
       where
         (v1First, v1Second) = VM.splitAt idx v1 -- this uses unsafeSlice, e.g. does not create a new vector
         (v2First, v2Second) = V.splitAt idx v2
         idx = len `div` 2
         len = VM.length v1
