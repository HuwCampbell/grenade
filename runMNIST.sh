stack=cabal
datadir=data
trainset=train-images-idx3-ubyte
trainlab=train-labels-idx1-ubyte
testset=t10k-images-idx3-ubyte
testlab=t10k-labels-idx1-ubyte
url=http://yann.lecun.com/exdb/mnist
exec="$stack exec mnist $datadir -- +RTS -N -s"
if [ -f $datadir/$trainset.gz ]
then
  $stack build mnist && $exec
 else
  mkdir -p data
  echo "Attempting to download MNIST data"
  curl -o $datadir/$trainset.gz $url/$trainset.gz
  curl -o $datadir/$trainlab.gz $url/$trainlab.gz
  curl -o $datadir/$testset.gz $url/$testset.gz
  curl -o $datadir/$testlab.gz $url/$testlab.gz

  if [ -f $datadir/$trainset.gz ]
   then
    $stack build mnist && $exec
   else
    echo "$datadir/$trainset does not exist. Please download MNIST files to $datadir/"
  fi
fi
