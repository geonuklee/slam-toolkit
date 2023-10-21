cwd=$PWD
pkg=${PWD##*/}
buildtype=RelWithDebInfo
#buildtype=Debug

cd build
cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=On -DCMAKE_BUILD_TYPE=$buildtype && cp compile_commands.json ~/.vim/ && make -j4
make -j4
retval=$?

#dataset="waymo"
#seq=segment-3015436519694987712_1300_000_1320_000_with_camera_labels # 적절한 주변 차량

dataset="kitti"
start_frame=0
seq="0000"
#seq="0001" # static
#seq="0002" # Dynamic 인식률 낮은 secene, keyframe 추가만 해줘도 인식률 훨씬 개선되겠다.
#seq="0003" # dynamic
#seq="0004" # dynamic
#seq="0005" # 고속도로 장면.
#seq="0006" # 정지한 화면에서는 움직이는 instance 에 추가적으로 mappoint 할당이 없어서, 물체 감지는 안된다.
#seq="0007" # ㄹ자 주행. 시작에서만 움직이는 차량
#seq="0008" # 고속도로장면, 조금 부정확.
#seq="0011" #
#seq="0013"
#seq="0014" # 우회전, 움직이는 차량.
#seq="0015" # 직진후 횡동보도 보행자.
#seq="0018"  # 직진중 선행차량, 맞은편 차량.
#seq="0019" # Tram
#seq="0020"


output_dir="output"

#msg "Build is done"
if [ $retval -eq 0 ]; then
  git log -1 > ../output/each_commit.txt
  git diff HEAD ../src/*.cpp >> ../output/each_commit.txt
  #./example_segmentation $seq $output_dir $start_frame
  gdb -ex "set breakpoint pending on" -ex "run" --args ./example_segmentation $seq $output_dir $start_frame
  #gdb -ex "set breakpoint pending on" -ex "b matcher.cpp:209" -ex "run" --args ./example_segmentation $seq
  #gdb -ex "set breakpoint pending on"  -ex "run" --args ./example_segmentation $seq
  #gdb -ex "set breakpoint pending on" -ex "b abort" -ex "run" --args ./example_segmentation $seq
  #gdb -ex "set breakpoint pending on" -ex "break abort "  -ex "run" --args ./example_segmentation
  #gdb -ex "b visualize.cpp:475" -ex "break abort "  -ex "run" --args ./example_segmentation $dataset $seq 0
fi
cd $cwd
#if [ $retval -eq 0 ]; then
#  python2 scripts/eval.py training_$seq
#fi
