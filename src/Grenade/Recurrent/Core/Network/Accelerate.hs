module Grenade.Recurrent.Core.Network.Accelerate where

-- | Type of a recurrent neural network.
--
--   The [*] type specifies the types of the layers.
--
--   The [Shape] type specifies the shapes of data passed between the layers.
--
--   The definition is similar to a Network, but every layer in the
--   type is tagged by whether it's a FeedForward Layer of a Recurrent layer.
--
--   Often, to make the definitions more concise, one will use a type alias
--   for these empty data types.
data RecurrentNetwork :: [*] -> [Shape] -> * where
  RNil   :: SingI i
         => RecurrentNetwork '[] '[i]

  (:~~>) :: (SingI i, Layer x i h)
         => !x
         -> !(RecurrentNetwork xs (h ': hs))
         -> RecurrentNetwork (FeedForward x ': xs) (i ': h ': hs)

  (:~@>) :: (SingI i, RecurrentLayer x i h)
         => !x
         -> !(RecurrentNetwork xs (h ': hs))
         -> RecurrentNetwork (Recurrent x ': xs) (i ': h ': hs)
infixr 5 :~~>
infixr 5 :~@>

-- | Gradient of a network.
--
--   Parameterised on the layers of the network.
data RecurrentGradient :: [*] -> * where
   RGNil  :: RecurrentGradient '[]

   (://>) :: UpdateLayer x
          => Gradient x
          -> RecurrentGradient xs
          -> RecurrentGradient (phantom x ': xs)

-- | Recurrent inputs (sideways shapes on an imaginary unrolled graph)
--   Parameterised on the layers of a Network.
data RecurrentInputs :: [*] -> * where
   RINil   :: RecurrentInputs '[]

   (:~~+>) :: (UpdateLayer x, Fractional (RecurrentInputs xs))
           => ()                      -> !(RecurrentInputs xs) -> RecurrentInputs (FeedForward x ': xs)

   (:~@+>) :: (Fractional (RecurrentShape x), Fractional (RecurrentInputs xs), RecurrentUpdateLayer x)
           => !(RecurrentShape x) -> !(RecurrentInputs xs) -> RecurrentInputs (Recurrent x ': xs)

-- | All the information required to backpropogate
--   through time safely.
--
--   We index on the time step length as well, to ensure
--   that that all Tape lengths are the same.
data RecurrentTape :: [*] -> [Shape] -> * where
   TRNil  :: SingI i
          => RecurrentTape '[] '[i]

   (:\~>) :: Tape x i h
          -> !(RecurrentTape xs (h ': hs))
          -> RecurrentTape (FeedForward x ': xs) (i ': h ': hs)

   (:\@>) :: RecTape x i h
          -> !(RecurrentTape xs (h ': hs))
          -> RecurrentTape (Recurrent x ': xs) (i ': h ': hs)


runRecurrent
  :: forall shapes layers.
     RecurrentNetwork layers shapes
  -> RecurrentInputs layers
  -> S (Head shapes)
  -> (RecurrentTape layers shapes, RecurrentInputs layers, S (Last shapes))
runRecurrent =
  go
    where
  go  :: forall js sublayers. (Last js ~ Last shapes)
      => RecurrentNetwork sublayers js
      -> RecurrentInputs sublayers
      -> S (Head js)
      -> (RecurrentTape sublayers js, RecurrentInputs sublayers, S (Last js))
  go (!layer :~~> n) (() :~~+> nIn) !x
      = let (!tape, !forwards) = runForwards layer x

            -- recursively run the rest of the network, and get the gradients from above.
            (!newFN, !ig, !answer) = go n nIn forwards
        in (tape :\~> newFN, () :~~+> ig, answer)

  -- This is a recurrent layer, so we need to do a scan, first input to last, providing
  -- the recurrent shape output to the next layer.
  go (layer :~@> n) (recIn :~@+> nIn) !x
      = let (tape, shape, forwards) = runRecurrentForwards layer recIn x
            (newFN, ig, answer) = go n nIn forwards
        in (tape :\@> newFN, shape :~@+> ig, answer)

  -- Handle the output layer, bouncing the derivatives back down.
  -- We may not have a target for each example, so when we don't use 0 gradient.
  go RNil RINil !x
    = (TRNil, RINil, x)
