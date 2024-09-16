# Bazel-6.5.0 RISC-V

## obs环境搭建

### 注册账号

https://build.tarsier-infra.isrc.ac.cn/

### 配置环境

在[链接](https://download.opensuse.org/repositories/openSUSE:/Tools/Fedora_Rawhide/noarch/)中下载obs-build-mkbaselibs和obs-build的rpm包

```
dnf install osc obs-build-mkbaselibs-20240723-455.110.noarch.rpm obs-build-20240723-455.110.noarch.rpm -y
vi ~/.oscrc
# 写入以下内容：
[general]
apiurl = https://build.tarsier-infra.isrc.ac.cn/
no_verify = 1

[https://build.tarsier-infra.isrc.ac.cn/]
user=username
pass=passwd
```

## Bazel rpm包构建

```
mkdir obs; cd obs
osc co home:6eanut:branches:home:6eanut/bazel
cd home\:6eanut\:branches\:home\:6eanut/bazel/
osc build
dnf install /var/tmp/build-root/bazel-riscv64/home/abuild/rpmbuild/RPMS/riscv64/bazel-6.5.0-1.oe2403.riscv64.rpm -y
bazel version
```
