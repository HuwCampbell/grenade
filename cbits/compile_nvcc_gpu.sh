
for f in *.cu; do
    nvcc -ptx $f
done
