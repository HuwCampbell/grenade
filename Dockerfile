# 
# NOTE: Tried with Haskell 8.2 and 8.0 and ended up with some build errors
#
FROM haskell:8.4

RUN mkdir /grenade
COPY . /grenade

# 
# Install system dependencies
# 
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    libatlas-base-dev \
    libgsl0-dev \
    liblapack-dev \
    python \
    wget

#
# Install Open Blas (with Lapack included)
# This is a depencency for building hmatrix package
#
ENV OPEN_BLAS_VER=0.2.20
RUN wget http://github.com/xianyi/OpenBLAS/archive/v$OPEN_BLAS_VER.tar.gz
RUN tar -xzvf v$OPEN_BLAS_VER.tar.gz

WORKDIR /OpenBLAS-$OPEN_BLAS_VER
RUN ls -la
RUN make FC=gfortran
RUN make PREFIX=/usr/local install

#
# Build and setup grenade
#
WORKDIR /grenade

RUN stack init
RUN stack setup --install-ghc
RUN stack build

# Optional GHCI prompt configuration
RUN echo ':set prompt "\ESC[34mλ> \ESC[m"' > ~/.ghci

#
# Choose default main object to run GHCI
#
ENV GRENADE_MAIN=mnist

#
# Run GHC repl
#
CMD stack ghci --main-is grenade-examples:exe:$GRENADE_MAIN