# ref : https://cloud.google.com/storage/docs/gsutil_install?hl=ko#deb
sudo apt-get -y install apt-transport-https ca-certificates gnupg curl
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
sudo apt-get update && sudo apt-get install google-cloud-cli

# 파이썬 설정
sudo apt-get install python3.8
sudo update-alternatives --list python3
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 20
sudo apt-get install --reinstal python3-pip
#pip3 install PyQt5 notebook # notebook for npyb files, qt for matplotlib
pip3 install --user waymo-open-dataset-tf-2-11-0 --upgrade
