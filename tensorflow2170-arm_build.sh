#!/bin/sh

set -e # 在遇到非零返回值时立即退出

start=$(date +%s)

# 创建python venv
/usr/local/python-3.11.6/bin/python3.11 -m venv venv00
source venv00/bin/activate
pip install -U pip numpy wheel
pip install -U keras_preprocessing --no-deps

echo "python venv创建成功"

# 下载bazel 二进制
mkdir install; cd install
wget https://github.com/bazelbuild/bazel/releases/download/6.5.0/bazel-6.5.0-linux-arm64
mv bazel-6.5.0-linux-arm64 bazel
chmod +x bazel
BAZEL=$(pwd)
PATH="$BAZEL:$PATH"
echo "# bazel" >> ~/.bashrc
echo "export BAZEL=$BAZEL" >> ~/.bashrc
echo "export PATH=$PATH" >> ~/.bashrc
source ~/.bashrc
cd ..

echo "bazel环境搭建成功"

# 构建tensorflow wheel
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
git checkout tags/v2.17.0
./configure
export TF_PYTHON_VERSION=3.11
bazel build //tensorflow/tools/pip_package:wheel --repo_env=WHEEL_NAME=tensorflow --local_ram_resources=600 --jobs=4
pip install bazel-bin/tensorflow/tools/pip_package/wheel_house/tensorflow-2.17.0-cp311-cp311-linux_aarch64.whl
cd ..

echo "tensorflow wheel构建成功并安装"

# 构建tf-text wheel
git clone https://github.com/tensorflow/text.git
cd text
git checkout tags/v2.17.0
./oss_scripts/run_build.sh
pip install tensorflow_text-2.17.0-cp311-cp311-linux_aarch64.whl
cd ..

echo "tensorflow-text wheel构建成功并安装"

# 下载tf-models-official wheel
wget https://files.pythonhosted.org/packages/38/79/18a7380e0f5f7961a03939e907d115eacb06253d8fbb4252cc4f0a09b642/tf_models_official-2.17.0-py2.py3-none-any.whl
pip install tf_models_official-2.17.0-py2.py3-none-any.whl

echo "tf-models-official wheel下载成功并安装"

end=$(date +%s)
runtime=$((end-start))
echo "脚本执行时长： $runtime s"