#!/bin/bash -eu 
example=iris
stack=cabal

dataset="iris.data"
datalab="iris.names"
url="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/"

datadir="data"
args=""
if [ $# -ge 1 ]
then
  case $1 in
    (-?*)
      echo >&2 "Missing initial data directory argument"
      exit 1;;
    (*)
      datadir=$1
      shift
      args=$*;;
  esac
fi

exec="$stack exec $example --RTS -- $datadir $args +RTS -N -s"

if [ -f "$datadir/$dataset" ]
then
  $stack build $example && $exec
 else
  mkdir -p "$datadir"
  echo "Attempting to download Iris data"
  curl -o "$datadir/$dataset" "$url/$dataset"
  curl -o "$datadir/$datalab" "$url/$datalab"

  if [ -f "$datadir/$dataset" ]
   then
    $stack build $example && $exec
   else
    echo "$datadir/$dataset does not exist. Please download Iris files to $datadir/"
  fi
fi
