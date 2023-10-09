cwd=$PWD
pkg=${PWD##*/}

#rm -rf output
buildtype=RelWithDebInfo
cd build
cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=On -DCMAKE_BUILD_TYPE=$buildtype && cp compile_commands.json ~/.vim/ && make -j4
make -j4
retval=$?


if [ $retval -eq 0 ]; then
    cp example_segmentation example_segmentation_batch
    cd $cwd
    output_dir="output_batch"
    rm -rf $ouput_dir
    mkdir $ouput_dir
    git log -1 > $output_dir/batch_commit.txt
    git diff HEAD src/*.cpp >> $output_dir/batch_commit.txt
    im_dir=kitti_tracking_dataset/training/image_02
    for dir in "$im_dir"/*; do
      if [ -d "$dir" ]; then
        seq=$(basename $dir)
        echo $seq
        ./build/example_segmentation_batch $seq $output_dir
      fi
    done
    #msg "Batch SLAM test is done. Now start eval..."
    python2 scripts/eval.py batch
    msg "Batch SLAM eval is done"
fi

