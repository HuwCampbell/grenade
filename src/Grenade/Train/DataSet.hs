module Grenade.Train.DataSet where

import Grenade.Core.Shape (S)

type DataPoint i o = (S i, S o)

type DataSet i o = [DataPoint i o]
