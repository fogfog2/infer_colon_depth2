# infer_colon_depth2

[model file download]

    https://drive.google.com/file/d/1iptlHzk3lVSEy__ix8ENuStUhiNeJXXE/view?usp=sharing
    ./models/model2.ckpt

[windows]

    conda create --name depthnet python=3.7

    conda activate depthnet

    pip install -r requirements.txt

    [pytorch install, https://pytorch.org/get-started/previous-versions/]

    ex cpu] conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cpuonly -c pytorch

    ex gpu] conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge

    [run-usbcam] python infer_colon_simple.py --checkpoint=./models/model2.ckpt --input=cam 

    [run-file] python infer_colon_simple.py --checkpoint=./models/model2.ckpt --input=./test.mp4


[ubuntu]

    packnet_env.yml 파일 prefix 경로 수정

    conda env create --file packnet_env.yml

    conda activate packnet_env

    sh run.sh 

    run.sh 파일 내부 수정법

    카메라 --input=cam 

    동영상 --input=/path/xxx.mp4

    이미지 --input=/path/folder
