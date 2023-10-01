cwd=$PWD
pkg=${PWD##*/}

im_dir=kitti_tracking_dataset/training/image_02     

for dir in "$im_dir"/*; do
  if [ -d "$dir" ]; then
    seq=$(basename $dir)
    echo $seq
    ./build/example_segmentation $seq
  fi
done

cd $cwd
