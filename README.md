# ProtoBully

基于原型匹配和多模态结构对齐的社交媒体网络霸凌检测系统。该项目采用图结构原型匹配技术，通过分析用户互动的图结构模式来检测霸凌行为。

## 项目结构

\`\`\`
ProtoBully/
 src/                           # 源代码目录
    data_processing/          # 数据预处理模块
       base_processor.py     # 基础数据处理器
       metadata_processor.py # 元数据处理
      ─ multimodal_processor.py # 多模态数据处理
       text_processor.py     # 文本数据处理
       video_processor.py    # 视频数据处理
   
    feature_extraction/       # 特征提取模块
       base_feature_extractor.py
       bert_comment_enhancer.py
       multimodal_feature_extractor.py
       text_feature_extractor.py
       user_feature_extractor.py
   
    graph_construction/       # 图构建模块
       base_graph_builder.py
       graph_construction_runner.py
       heterogeneous_graph_builder.py
   
    multimodal-aligner/      # 多模态对齐模块
       multimodal_structure_aligner_v4_optimized.py
   
    prototype/               # 原型提取和检测模块
       bert_emotion_analyzer.py
       bullying_subgraph_filter.py
       cyberbullying_detector_v6.py
       prototype_extractor_v6_refactored.py
   
    utils/                   # 工具类
        bert_emotion_analyzer.py
        config.py
        logger.py

## 运行说明

1. 数据预处理：
python src/data_processing/main.py

2. 特征提取：
python src/feature_extraction/text_feature_extractor.py

3. 图构建：
python src/graph_construction/graph_construction_runner.py

4. 子图提取：
python run_fixed_subgraph_extraction.py
输出目录：data/subgraphs/universal_optimized_fixed/

5. 原型提取（使用最新重构版本）：
python run_prototype_extraction_v6_refactored.py
输出文件：data/prototypes/extracted_prototypes_v6_refactored_20250703_092014.pkl

6. 霸凌检测：
python run_cyberbullying_detection_v6_adapted.py

## 数据目录说明

- data/graphs/: 异构图数据
- data/processed/: 处理后的数据
- data/prototypes/: 提取的原型
- data/models/: 训练的模型
- data/subgraphs/: 提取的子图数据

## 依赖环境

- Python 3.8+
- PyTorch
- DGL (Deep Graph Library)
- BERT
- Scikit-learn
- NumPy & Pandas

## 注意事项

1. 运行顺序必须按照上述步骤顺序执行
2. 确保data目录下有必要的输入数据
3. 大型数据文件未包含在代码仓库中，需单独获取
