---
title: github 以及 git 的简单配置
---
对于存放代码的工具，我们使用最多的应该就是 [github](https://github.com/)! 
我们可以使用它来存放我们的代码，同时也可以看到许多大神们开源的项目 ，具体怎么查找就先不在这里展示了，
相信大家会根据自己的需要去搜索的, 
如果我们想要更好的使用 [github](https://github.com/)! 就需要在我们的本地配置 [git](https://git-scm.com/downloads) 工具! 
这里重点记录下自己是如何配置 git 部分的。


### 注册一个 github 账号

这一步比较简单，直接使用邮箱注册即可，一般什么邮箱都可以，qq邮箱，163邮箱，gmail邮箱均可
注册地址: [Sign up](https://github.com/signup?ref_cta=Sign+up&ref_loc=header+logged+out&ref_page=%2F&source=header-home)
如果已经有账号密码的话，可以直接登录
登录地址：[Sign in](https://github.com/login)

### 安装 git (基于linux环境ubuntu系统)

直接使用 ubuntu 系统自带的 apt 安装工具进行安装
安装命令如下：
``` bash
$ sudo apt install git -y
```
补充下：如果是 Centos 系统，则只需将这里的 apt 更换为 yum 即可
验证是否安装成功命令：
``` bash
$ git --version
```
如果返回版本号，则说明安装成功，可以正常使用


### 配置 git config 部分

安装在 ubuntu 环境上，git可能需要依赖的三方库或系统插件
``` bash
$ apt install build-essential nghttp2
$ apt install libnghttp2-dev
$ apt install libssl-dev
```

More info: [Generating](https://hexo.io/docs/generating.html)

### Deploy to remote sites

``` bash
$ hexo deploy
```

More info: [Deployment](https://hexo.io/docs/one-command-deployment.html)
