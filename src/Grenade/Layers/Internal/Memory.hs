{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE ViewPatterns #-}

module Grenade.Layers.Internal.Memory
    ( withTempVector
    , releaseTmpVectors
    , UUID
    , newUUID
    ) where

import           Control.Concurrent
import           Control.Concurrent.MVar
import           Control.DeepSeq
import           Control.Monad                (void)
import           Data.IORef
import           Data.List                    (foldl')
import qualified Data.Map.Strict              as M
import qualified Data.Vector.Storable         as V
import           System.IO.Unsafe             (unsafePerformIO)
import           System.Random.MWC

import           Grenade.Layers.Internal.BLAS
import           Grenade.Types
import           Grenade.Utils.Vector

import           Debug.Trace

type UUID = Int

newUUID :: UUID
newUUID = unsafePerformIO $ withSystemRandom $ asGenIO $ uniformR (minBound, maxBound)
{-# NOINLINE newUUID #-}

tempVars :: IORef (M.Map (UUID, Int) [V.Vector RealNum], M.Map (UUID, Int) [V.Vector RealNum])
tempVars = unsafePerformIO $ newIORef mempty
{-# NOINLINE tempVars #-}

uuidMap :: IORef (M.Map UUID [Int])
uuidMap = unsafePerformIO $ newIORef mempty
{-# NOINLINE uuidMap #-}

addMap :: UUID -> Int -> IO ()
addMap uuid size = modifyIORef uuidMap $ M.alter add uuid
  where
    add Nothing = Just [size]
    add c@(Just xs)
      | size `elem` xs = c
      | otherwise = Just (size : xs)

forwardCount :: IORef Int
forwardCount = unsafePerformIO $ newIORef 0
{-# NOINLINE forwardCount #-}


releaseTmpVectors :: UUID -> ()
releaseTmpVectors uuid =
  unsafePerformIO $ do
    uuids <- readIORef uuidMap
    let sizes = M.findWithDefault (error "empty ") uuid uuids
    writeIORef forwardCount 0
    atomicModifyIORef tempVars $ \(free, locked) ->
      ( ( foldl' (\mFree size -> M.insertWith (++) (uuid, size) (M.findWithDefault [] (uuid, size) locked) mFree) free sizes
        , foldl' (\m size -> M.delete (uuid, size) m) locked sizes)
      , ())


withTempVector :: UUID -> Int -> (V.Vector RealNum -> IO (V.Vector RealNum)) -> V.Vector RealNum
withTempVector _    size f | size < 16 = unsafePerformIO $ createVector size >>= f -- don't reuse small vectors
withTempVector uuid size f =
  unsafePerformIO $ do
    n <- readIORef forwardCount
    writeIORef forwardCount (n + 1)
    addMap uuid size
    let lock = n < 25
    tmp <-
      atomicModifyIORef tempVars $ \ms@(free, locked) ->
        case M.lookup key free of
          Nothing -> new ms
          Just [] -> new ms
          Just (x:xs)
            | lock -> ((M.insert key xs free, M.insertWith (++) key [x] locked), x)
          Just (x:xs) -- only forward runs
            | n < 30 -> ((M.insert key (xs ++ M.findWithDefault [] key locked ++ [x]) free, M.insert key [] locked), x)
          Just (x:xs) -> ((M.insert key (xs ++ [x]) free, locked), x)
    f tmp
  where
    key = (uuid, size)
    maxSize = 32 -- Maximum size of list of temporary vectors
    insMax xs x
      | length xs >= maxSize = xs -- Ensure that we don't put too many into the list
      | otherwise = (xs ++ x)
    new (free, locked) = ((free, M.insertWith insMax key [x] locked), x)
      where
        x = createVectorUnsafe size
{-# NOINLINE withTempVector #-}
