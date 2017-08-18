{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE FlexibleInstances     #-}

-- I'm not sure if I need this, but couldn't figure out how to avoid using it
{-# LANGUAGE UndecidableInstances  #-}

{-|
Module      : Grenade.Core.Network
Description : Core definition of a Neural Network
Copyright   : (c) Huw Campbell, 2016-2017
License     : BSD2
Stability   : experimental

This module defines the core data types and functions
for non-recurrent neural networks.
-}

module Grenade.Core.Network.Accelerate (
    {-# Network (..)
  , #-} Gradients (..)
  , Tapes (..)

  , runNetwork
  , runGradient
  -- , applyUpdate

  -- , randomNetwork
  ) where

import           Data.Array.Accelerate
import           Data.Singletons
import           Data.Singletons.Prelude.List
import           Data.Singletons.TypeLits

import           Grenade.Core.Layer.Accelerate
import           Grenade.Core.LearningParameters.Accelerate
import qualified Grenade.Core as G
import           Grenade.Core.Shape.Accelerate
import           Grenade.Core.Shape (Shape)
import           Grenade.Core.Matrix.Accelerate

-- | Type of a network.
--
--   The @[*]@ type specifies the types of the layers.
--
--   The @[Shape]@ type specifies the shapes of data passed between the layers.
--
--   Can be considered to be a heterogeneous list of layers which are able to
--   transform the data shapes of the network.
-- data Network :: [*] -> [G.Shape] -> * where
--     NNil  :: SingI i
--           => Network '[] '[i]

--     (:~>) :: (SingI i, SingI h, G.Layer l i h, Accelerable l)
--           => !(Accelerated l)
--           -> !(Accelerated (G.Network ls (h ': hs)))
--           -> Accelerated (G.Network (l ': ls) (i ': h ': hs))
-- infixr 5 :~>

-- instance Show (Network n) where
--   show NNil = "NNil"
-- instance (Show x, Show (Network xs rs)) => Show (Network (x ': xs) (i ': rs)) where
--   show (x :~> xs) = show x ++ "\n~>\n" ++ show xs


instance Accelerable (G.Network '[] '[i]) where
  data Accelerated (G.Network '[] '[i]) = ANil
  toAccel G.NNil = ANil

instance
  ( Accelerable l
  , Accelerable (G.S i)
  , Accelerable (G.S h)
  , Accelerable (G.Network ls (h ': hs))
  )
  => Accelerable
    (G.Network (l ': ls) (i ': h ': hs)) where

  data Accelerated (G.Network (l ': ls) (i ': h ': hs)) =
    ANetwork (Accelerated l) (Accelerated (G.Network ls (h ': hs)))

  toAccel (x G.:~> xs) = ANetwork (toAccel x) (toAccel xs)


-- | Gradient of a network.
--
--   Parameterised on the layers of the network.
data Gradients :: [*] -> * where
   GNil  :: Gradients '[]

   (:/>) :: UpdateLayer l
         => Gradient l
         -> Gradients ls
         -> Gradients (l ': ls)

-- | Wegnert Tape of a network.
--
--   Parameterised on the layers and shapes of the network.
data Tapes :: [*] -> [G.Shape] -> * where
   TNil  :: SingI i
         => Tapes '[] '[i]

   (:\>) :: (SingI i, SingI h, Layer l i h, Accelerable l)
         => !(Tape l i h)
         -> !(Tapes ls (h ': hs))
         -> Tapes (l ': ls) (i ': h ': hs)


-- | Running a network forwards with some input data.
--
--   This gives the output, and the Wengert tape required for back
--   propagation.
runNetwork :: forall layers shapes h l.
              ( Accelerable (G.S (Head shapes))
              , Accelerable (G.S (Last shapes))
              )
           => Accelerated (G.Network layers shapes)
           -> Accelerated (G.S (Head shapes))
           -> (Tapes layers shapes, Accelerated (G.S (Last shapes)))
runNetwork = undefined
--   go
--     where
--   go  :: forall js ss. (Last js ~ Last shapes)
--       => Network ss js
--       -> S (Head js)
--       -> (Tapes ss js, S (Last js))
--   go (layer :~> n) !x =
--     let (tape, forward) = runForwards layer x
--         (tapes, answer) = go n forward
--     in  (tape :\> tapes, answer)

--   go NNil !x
--       = (TNil, x)


-- | Running a loss gradient back through the network.
--
--   This requires a Wengert tape, generated with the appropriate input
--   for the loss.
--
--   Gives the gradients for the layer, and the gradient across the
--   input (which may not be required).
runGradient :: forall layers shapes h l.
              ( Accelerable (G.S (Head shapes))
              , Accelerable (G.S (Last shapes))
              )
            => Accelerated (G.Network layers shapes)
            -> Tapes layers shapes
            -> Accelerated (G.S (Last shapes))
            -> (Gradients layers, Accelerated (G.S (Head shapes)))
runGradient net tapes o = undefined
--   go net tapes
--     where
--   go  :: forall js ss. (Last js ~ Last shapes)
--       => Network ss js
--       -> Tapes ss js
--       -> (Gradients ss, S (Head js))
--   go (layer :~> n) (tape :\> nt) =
--     let (gradients, feed)  = go n nt
--         (layer', backGrad) = runBackwards layer tape feed
--     in  (layer' :/> gradients, backGrad)

--   go NNil TNil
--       = (GNil, o)


-- -- | Apply one step of stochastic gradient decent across the network.
-- applyUpdate :: LearningParameters
--             -> Network layers shapes
--             -> Gradients layers
--             -> Network layers shapes
-- applyUpdate rate (layer :~> rest) (gradient :/> grest)
--   = runUpdate rate layer gradient :~> applyUpdate rate rest grest

-- applyUpdate _ NNil GNil
--   = NNil


-- -- | Ultimate composition.
-- --
-- --   This allows a complete network to be treated as a layer in a larger network.
-- instance UpdateLayer (GN.Network sublayers subshapes) (Network asublayers asubshapes) where
--   type Gradient (Network asublayers asubshapes) = Gradients asublayers
--   runUpdate    = applyUpdate

-- -- | Ultimate composition.
-- --
-- --   This allows a complete network to be treated as a layer in a larger network.
-- instance (i ~ (Head subshapes), o ~ (Last subshapes))
--   -- => Layer (GN.Network sublayers subshapes) (Network asublayers asubshapes) i o where

--   type Tape (Network asublayers asubshapes) i o = Tapes asublayers asubshapes
--   runForwards  = runNetwork
--   runBackwards = runGradient
