cwd=$PWD
pkg=${PWD##*/}

buildtype=RelWithDebInfo
cd build
cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=On -DCMAKE_BUILD_TYPE=$buildtype && cp compile_commands.json ~/.vim/ && make -j4
make -j4
retval=$?

git diff > output/diff.txt

if [ $retval -eq 0 ]; then
    cd $cwd
    im_dir=kitti_tracking_dataset/training/image_02
    for dir in "$im_dir"/*; do
      if [ -d "$dir" ]; then
        seq=$(basename $dir)
        echo $seq
        ./build/example_segmentation $seq
      fi
    done
    python2 scripts/eval.py batch
    msg "Batch SLAM test/eval is done"
fi

