#!/bin/sh

PACKAGES=( "finetune" )

if [ -d ".wheels" ]
then
  echo "wheels exists..."
else
  echo "wheels doesnt exist creating..."
  mkdir ".wheels"
fi

for PACKAGE in "${PACKAGES[@]}"
do
  cd $PACKAGE && \
  python setup.py bdist_wheel && \
  cp dist/*.whl ../.wheels/
  rm -rf "$PACKAGE.egg-info" dist  build && \
  cd ..
done


docker build -t "amj3d1/wav2vec2-finetune-custom" .