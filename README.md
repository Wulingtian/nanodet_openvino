nanodet简介

NanoDet （https://github.com/RangiLyu/nanodet）是一个速度超快和轻量级的Anchor-free 目标检测模型

环境配置

Ubuntu：18.04

OpenVINO：2020.4

OpenCV：3.4.2

OpenVINO和OpenCV安装包（编译好了，也可以自己从官网下载编译）可以从链接: https://pan.baidu.com/s/1zxtPKm-Q48Is5mzKbjGHeg 密码: gw5c下载

OpenVINO安装

tar -xvzf l_openvino_toolkit_p_2020.4.287.tgz

cd l_openvino_toolkit_p_2020.4.287

sudo ./install_GUI.sh 一路next安装

cd /opt/intel/openvino/install_dependencies

sudo -E ./install_openvino_dependencies.sh

vim ~/.bashrc

把如下两行放置到bashrc文件尾

source /opt/intel/openvino/bin/setupvars.sh

source /opt/intel/openvino/opencv/setupvars.sh

source ~/.bashrc 激活环境

模型优化配置步骤

cd /opt/intel/openvino/deployment_tools/model_optimizer/install_prerequisites

sudo ./install_prerequisites_onnx.sh（模型是从onnx转为IR文件，只需配置onnx依赖）

OpenCV配置

tar -xvzf opencv-3.4.2.zip 解压OpenCV到用户根目录即可，以便后续调用
NanoDet模型转换

pip install onnx

pip install onnx-simplifier

git clone https://github.com/Wulingtian/nanodet.git

cd nanodet

cd config 配置模型文件，训练模型

定位到nanodet目录，进入tools目录，打开export.py文件，配置cfg_path model_path out_path三个参数

定位到nanodet目录，运行 python tools/export.py 得到转换后的onnx模型

python3 -m onnxsim onnx模型名称 nanodet-simple.onnx 得到最终简化后的onnx模型

python /opt/intel/openvino/deployment_tools/model_optimizer/mo_onnx.py --input_model onnx简化的模型 --output_dir 期望模型输出的路径；得到IR文件
NanoDet模型部署

sudo apt install cmake 安装cmake

git clone https://github.com/Wulingtian/nanodet_openvino.git （求star！）

cd nanodet_openvino 打开CMakeLists.txt文件，修改OpenCV_INCLUDE_DIRS和OpenCV_LIBS_DIR，之前已经把OpenCV解压到根目录了，所以按照你自己的路径指定

定位到nanodet_openvino，cd models 把之前生成的IR模型（包括bin和xml文件）文件放到该目录下

定位到nanodet_openvino， cd test_imgs 把需要测试的图片放到该目录下

定位到nanodet_openvino，编辑main.cpp，xml_path参数修改为"../models/你的模型名称.xml"

编辑 num_class 设置类别数，例如：我训练的模型是安全帽检测，只有1类，那么设置为1

编辑 src 设置测试图片路径，src参数修改为"../test_imgs/你的测试图片"

定位到nanodet_openvino

mkdir build
cd build
cmake ..
make

./detect_test 输出平均推理时间，以及保存预测图片到当前目录下，至此，部署完成！
