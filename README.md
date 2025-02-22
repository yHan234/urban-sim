## Dataset

"year" 文件夹，其中包含以 "land_{year}.til" 命名的每年土地数据，1 表示城市用地，0 表示非城市用地，-1 表示无需计算区域。

"{spatial}.tif" 文件，各种空间数据，32 位浮点数归一化到 [0, 1]。

## Reference

https://github.com/kwtk86/UrbanM2M

https://github.com/ndrplz/ConvLSTM_pytorch

## 任务书

> # 基于注意力机制和卷积长短期记忆网络的长时序城市扩张模拟方法研究
>
> ## 一、选题背景
>
> 城市扩张作为地球系统运转中的重要模拟场景之一，更精确、更真实、更智能地预测未来城市空间布局对国土空间数智治理和土地利用可持续发展至关重要。针对主流元胞自动机模型无法准确捕捉长时序城市扩张过程时空特征的问题，结合卷积长短时记忆网络和注意力机制构建城市扩张元胞自动机模型，实现多尺度时空邻域特征的挖掘，有利于提高模型的模拟预测精度，推动大数据、人工智能等先进技术在自然资源系统性复杂性问题研究上的创新应用。
>
> ## 二、内容与要求
>
> 1. 查阅ConvLSTM、注意力机制与城市土地利用变化、生态环境变化模拟等不同领域应用的相关文献12篇以上，翻译其中的2篇外文文献，按要求撰写文献综述和开题报告；
> 2. 构建基于注意力机制和ConvLSTM的长时序城市扩张模型，包括：
>    - 获取目标区域内空间变量栅格和时间序列城市用地栅格，并构建数据集；
>    - 获取训练样本集，利用卷积长短时记忆网络，引入注意力机制构建扩张模型；
>    - 模型训练，并基于训练完成的扩张模型进行预测，得到城市发展概率图；
>    - 元胞自动机模块设计，以及模型性能评估和消融实验；
> 3. 使用Python实现算法，以环杭州湾城市群为例开展精度验证，其中FoM指数达到0.15以上，精度优于传统方法；
> 4. 按要求完成毕业设计任务，撰写规范的毕业设计文档（论文）。
>
> ## 三、主要参考文献
>
> - [1] He JL, Xia L, Yao Y, Ye H and Zhang JB. Mining transition rules of cellular automata for simulating urban expansion by using the deep learning techniques[J]. International Journal of Geographical Information Systems, 2018, 32(10):2076-2097.
> - [2] Zhou ZH, Chen YM, Wang ZS and Lu FD. Integrating cellular automata with long short-term memory neural network to simulate urban expansion using time-series data[J]. International Journal of Applied Earth Observation and Geoinformation, 2024,103676.
> - [3] Zhou ZH, Chen YM, Liu XP, Zhang XC and Zhang HH. A maps-to-maps approach for simulating urban land expansion based on convolutional long short-term memory neural networks[J]. International Journal of Geographical Information Systems, 2024, 38(3).
