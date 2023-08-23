## aidb-qt-demo

AiDB Qt demo.

<p align="center">
 <img src="./doc/Qt_face.gif" width="480px"/>
 <img src="./doc/Qt_ocr.gif"  width="480px"/>
<p align="center">

### Quick Start


* Compile AiDB library(custom backend)

```asm
git clone https://github.com/TalkUHulk/ai.deploy.box.git
cd ai.deploy.box
mkdir build && cd build
cmake -DENGINE_ORT=ON -DENGINE_MNN=ON -DENGINE_NCNN=ON  -DENGINE_TNN=OFF -DENGINE_OPV=OFF -DENGINE_PPLite=OFF -DENGINE_NCNN_WASM=OFF -DBUILD_SAMPLE=OFF ../
make -j8
```

generate **libAiDB.so** in ai.deploy.box/build/source/


* Clone this rep

```asm
git clone https://github.com/TalkUHulk/aidb_qt_demo.git
```

* Prepare files demo need.

* place `libAiDB.so` in [aidb/lib](./aidb/lib)
* place [AIDBData.h](https://github.com/TalkUHulk/ai.deploy.box/blob/main/source/core/AIDBData.h) [AIDBDefine.h](https://github.com/TalkUHulk/ai.deploy.box/blob/main/source/core/AIDBDefine.h) [Interpreter.h](https://github.com/TalkUHulk/ai.deploy.box/blob/main/source/core/Interpreter.h) [Utility.h](https://github.com/TalkUHulk/ai.deploy.box/blob/main/source/utility/Utility.h) [face_align.h](https://github.com/TalkUHulk/ai.deploy.box/blob/main/source/utility/face_align.h) [td_obj.h](https://github.com/TalkUHulk/ai.deploy.box/blob/main/source/utility/td_obj.h) in [aidb/include](./aidb/include)
* download [models](https://github.com/TalkUHulk/ai.deploy.box/releases/download/1.0.0/models-lite.zip) and unzip in aidb_qt_demo that named models, and place [ai.deploy.box/config](https://github.com/TalkUHulk/ai.deploy.box/tree/main/config) 、[ai.deploy.box/extra](https://github.com/TalkUHulk/ai.deploy.box/tree/main/extra) in [config](./config) 、[extra](./extra)


* Compile Qt Demo

```asm
cd ai.deploy.box
mkdir build && cd build
cmake .. && make -j2
```



**folder structure**
```
aidb_qt_demo/
├── main.cpp
├── themes
├── resource
├── src
├── 3rdparty
├── aidb
    ├── include
    ├── lib
├── config
    ├── mnn_config.yaml
    ├── ncnn_config.yaml
    .
    .
    .
    └── onnx_config.yaml
├── models
    ├── onnx
    ├── ncnn
    ├── mnn
    ├── tnn
    ├── paddle
    └── openvino   
.
.
.
└── extra
    ├── ppocr_keys_v1.txt
    .
    .
    └── imagenet-1k-id.txt
```



