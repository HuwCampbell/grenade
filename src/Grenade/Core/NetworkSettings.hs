module Grenade.Core.NetworkSettings
    ( NetworkSettings(..)

    ) where


newtype NetworkSettings = NetworkSettings
  { setDropoutActive :: Bool
  }
