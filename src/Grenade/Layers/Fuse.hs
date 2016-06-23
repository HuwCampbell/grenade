{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE KindSignatures        #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE StandaloneDeriving    #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE PolyKinds             #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE FlexibleInstances     #-}


module Grenade.Layers.Fuse (
    Fuse (..)
  ) where

import           Grenade.Core.Network
import           Grenade.Core.Shape

-- | Fuse two layers into one layer.
--   This can be used to simplify a network if a complicated repeated structure is used.
--   This does however have a trade off, internal incremental states in the Wengert tape are
--   not retained during reverse accumulation. So less RAM is used, but more compute is required.
data Fuse :: (* -> *) -> Shape -> Shape -> Shape -> * where
    (:$$) :: (Show x, Show y, Layer m x i h, Layer m y h o, KnownShape h, KnownShape i, KnownShape o)
          => !x
          -> !y
          -> Fuse m i h o
infixr 5 :$$

instance Show (Fuse m i h o) where
  show (x :$$ y) = "(" ++ show x ++ " :$$ " ++ show y ++ ")"

instance (Monad m, KnownShape i, KnownShape h, KnownShape o) => Layer m (Fuse m i h o) i o where
  runForwards (x :$$ y) input = do
    yInput  :: S' h <- runForwards x input
    runForwards y yInput

  runBackards rate (x :$$ y) input backGradient = do
    yInput  :: S' h <- runForwards x input
    (y', yGrad)     <- runBackards rate y yInput backGradient
    (x', xGrad)     <- runBackards rate x input yGrad
    return (x' :$$ y', xGrad)
