cwd=$PWD
pkg=${PWD##*/}

#rm -rf output
buildtype=RelWithDebInfo
cd build
cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=On -DCMAKE_BUILD_TYPE=$buildtype && cp compile_commands.json ~/.vim/ && make -j4
make -j4
retval=$?


PID_FIRST=0

#source test example_segmentation
if [ $retval -eq 0 ]; then
  cp example_segmentation example_segmentation_batch$DISPLAY
  cd $cwd
  output_dir="output_batch_"$DISPLAY
  rm -rf $output_dir
  mkdir $output_dir
  ffmpeg -video_size 1420x970 -framerate 5 -f x11grab -i $DISPLAY -c:v libxvid -qscale:v 3 $output_dir/output.avi < /dev/null &
  PID=$!
  git log -1 > $output_dir/batch_commit.txt
  git diff HEAD src/*.cpp >> $output_dir/batch_commit.txt
  im_dir=kitti_tracking_dataset/training/image_02
  for dir in "$im_dir"/*; do
    if [ -d "$dir" ]; then
      seq=$(basename $dir)
      #if [ "$seq" = "0004" ]; then
      #  continue  # Pose tracker issue
      #fi
      #msg "Start seq $seq"
      ./build/example_segmentation_batch$DISPLAY $seq $output_dir
    fi
  done
  kill $PID
  msg $DISPLAY" Batch SLAM test is done. Now start eval..."
  python2 scripts/eval.py $output_dir
fi

