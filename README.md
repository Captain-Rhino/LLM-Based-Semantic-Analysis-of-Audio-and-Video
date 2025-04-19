| 任务：  **1.****考察各类** **AI** **大模型，挑选适合音视频语义分析的，进行优化使其契合音视频数据特点。**  **2.****对音频降噪、格式转换等，对视频解码、抽帧等，提升数据质量便于分析。**  **3.****利用模型从音频提取语音内容、情感等特征，从视频提取场景、物体等信息，构建融合音视频语义特征的分析模型，用大量标注数据训练优化。** |
| ------------------------------------------------------------ |
| 预期成果：  **1.****完成一篇关于音视频语义特征提取方法的毕业论文。**  **2.****实现所提出的音视频语义特征提取方法，展示模型对不同音视频数据的分析结果，包括音频和视频的语义理解、情感倾向、事件描述等。**  **3.****为音视频语义化分析在实际应用中的场景（如视频内容审核、智能客服、多媒体教育等）提供应用建议和优化方案。** |

### 当前进度&关键模块

#### Whisper

这里使用whisper进行语音转文本，并记录每一段语音对应的时间，作为一个json文件保存（transcription.json）

#### Opencv

这里是在使用传统方法（比如灰度图像素差值）进行关键帧的提取，保存到K_frame文件夹，同时keyframes_info.json文件里面有其时间信息

#### 大模型 like llama3.2-vision / deepseek:r1

将图片＋文字放到大模型进行处理，带记忆的，联系上下文，从而理解到整个视频

这里又分为**1) localhost的api**和 **2)云端提供的api(比如deepseekr1，百万token2块钱)**这个后面发现不太行，deepseek 只支持图片ocr，人物和背景都看不到



**现在已经做了**

**1.语音识别和json存储，2.关键帧提取和json存储关键帧信息，3.localhost部署llama3.2vision进行图像分析**

4.测试了deepseek图像分析，不行。   通义千问和gpt图像分析，可行



#### 后面需要做的工作

##### 情感倾向分析

找一些情感分析模型

##### 提取关键帧的其他方法

比如1)用ai的方法来做，kmeans聚类, 2)[ffmpeg进行关键帧提取](https://blog.csdn.net/justloveyou_/article/details/88076675) 3)使用CNN来保存图片在使用聚类法提取关键帧

##### 文+图+情感 联合调试

所有东西加到一起，读json文件的文本内容&关键帧信息+关键帧图像，用api调用大模型



#### 一些问题

##### 1.关键帧提取(灰度图像素差值方法)

容易漏掉关键帧

##### 2.whisper语音识别

识别基本准确，但是有些不符合中文语境，比如“建言献策”识别成了“见言献策”

##### 3.llama3.2-vision对于中文ocr的识别效果很一般

考虑使用gpt/通义千问



### 语音部分

这里最终决定使用sensevoice来识别语音,需要一步提取语音文件

sensevoice本地服务器保持常开，处理60s音频的时间为1.6s，识别结果为语音开始时间，结束时间，文本内容，保存为json文件

### 文图对照部分

这里实现了CN_CLIP根据文本描述进行图像查询，从而找到对应的关键帧，并对长句进行裁剪，根据字数的多少决定这段语音中关键帧提取的多少。



语义对齐： 在语义层面上实现不同模态之间的对齐，需要深入理解数据的潜在语义联系。
图神经网络（GNN）：在构建图像和文本之间的语义图时，利用GNN学习节点（模态数据）之间的语义关系，实现隐式的语义对齐。

预训练语言模型与视觉模型结合：如CLIP（Contrastive Language-Image Pre-training），通过对比学习在大量图像-文本对上训练，使模型学习到图像和文本在语义层面上的对应关系，实现高效的隐式语义对齐


##### 问题在于，没有考虑纯视频的输入，没有文字，也就抽不出来关键帧。。。

### 大模型语义识别部分

这里使用了qwen-vl-plus-latest，通义千问大规模视觉语言模型。大幅提升细节识别能力和文字识别能力，支持超百万像素分辨率和任意长宽比规格的图像。在广泛的视觉任务上提供卓越的性能，调用了**API**，实现了上下文联系，最后形成总结
