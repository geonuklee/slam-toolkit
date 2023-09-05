cwd=$PWD
pkg=${PWD##*/}
buildtype=RelWithDebInfo
#buildtype=Debug

cd build
cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=On -DCMAKE_BUILD_TYPE=$buildtype && cp compile_commands.json ~/.vim/ && make -j4
make -j4
retval=$?

seq=segment-3015436519694987712_1300_000_1320_000_with_camera_labels # 적절한 주변 차량 
#seq=segment-4013125682946523088_3540_000_3560_000_with_camera_labels
    #-ex "b segslam.cpp:1380" \
    #-ex "b segslam.cpp:640 if (ins->pth_==14 && qth==1)" \

if [ $retval -eq 0 ]; then
  #./example_segmentation waymo $seq 0
  gdb -ex "catch throw " \
    -ex "run" --args ./example_segmentation waymo $seq 0
fi

cd $cwd
#
