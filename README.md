# README #

# Install

```bash
sudo apt-get install libboost1.65-all-dev  # all file system only?
sudo apt install libflann-dev libvtk6-qt-dev qt5-default
# libcanberra-gtk-module libcanberra-gtk3-module

```
## ORBvoc.txt
cd thirdparty
wget https://github.com/raulmur/ORB_SLAM2/raw/master/Vocabulary/ORBvoc.txt.tar.gz
tar -xzf ORBvoc.txt.tar.gz

## Trouble shoot
* /usr/lib/libgl.so not found.
  ```bash
    cd /usr/lib;
    sudo ln -s x86_64-linux-gnu/libGL.so.1 ./
  ```

## gtest installation.
```bash
cd ~
git clone https://github.com/google/googletest.git
cd googletest
mkdir build && cd build
cmake .. -DBUILD_SHARED_LIBS=ON -DINSTALL_GTEST=ON -DCMAKE_INSTALL_PREFIX:PATH=/usr/local
make -j8
sudo make install
sudo ldconfig
```
