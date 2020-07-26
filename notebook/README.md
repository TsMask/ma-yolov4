# 开发环境模型调试

在开发环境 notebook 中将目录中的 `model目录/V版本`，`notebook目录` 进行 Sync Obs 同步。

- `model目录/V版本` 中可能含有多个版本，只需要选择需要调试的版本目录进行同步即可
- `notebook目录` 中含 `detect_image.py文件` 和 `detect_video.py文件` 是便于自己本机运行和在开发环境 `Terminal - TensorFlow-2.1.0` 中执行。

## 终端执行方式

1. 打开 `Terminal` 后命令，可以看到已同步的 `model` 和 `notebook`

```shell
cd work

# V0015 是你同步的模型版本
ls -lh model/V0015/model

ls -lh notebook
```

2. 先切换到 `tf-2.1.0` 环境

```shell
source /home/ma-user/anaconda3/bin/activate TensorFlow-2.1.0
# (TensorFlow-2.1.0) sh-4.3$
```

3. 选择执行你需要的识别 py

```shell
# 图片识别 detect_image.py 文件
python notebook/detect_image.py --image notebook/test.jpg --min_score 0.3 --show_box_label true --input_size 608 --version V0015

# 视频识别 detect_video.py 文件
python notebook/detect_video.py --video notebook/test.mp4 --min_score 0.3 --show_box_label true --input_size 608 --version V0015
```

**输入参数**

|      名称      |                        说明                        |
| :------------: | :------------------------------------------------: |
|     image      |                    图片文件路径                    |
|     video      |                    视频文件路径                    |
|   min_score    |               最小显示分数，默认 0.2               |
| show_box_label |         结果图是否显示框和文字，默认 true          |
|   input_size   | 图片尺寸统一处理大小识别 320/480/512/608，默认 608 |
|    version     |         选择你同步的模型版本号，默认 V0xxx         |
