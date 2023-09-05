cwd=$PWD
pkg=${PWD##*/}
buildtype=RelWithDebInfo
seq_dir=../thirdparty/waymo-dataset/output/sequences
cd build
cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=On -DCMAKE_BUILD_TYPE=$buildtype && cp compile_commands.json ~/.vim/ && make -j4
make -j4
retval=$?

if [ $retval -eq 0 ]; then
  for seq in $(ls ../$seq_dir); do
    echo $seq
    ./example_segmentation waymo $seq 1
    retval=$?
    echo $retval
    if [ $retval != '1' ]; then
      gdb -ex "catch throw " \
        -ex "run" --args ./example_segmentation waymo $seq 1
    fi
  done
fi

cd $cwd
