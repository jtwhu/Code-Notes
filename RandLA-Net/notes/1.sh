G:.
###
 # @Descripttion: 
 # @Author: Jiang Tao
 # @version: 
 # @Date: 2022-08-06 10:01:21
 # @LastEditors: Jiang Tao
 # @LastEditTime: 2022-08-06 11:23:22
### 
│  .gitattributes
│  .gitignore
│  compile_op.sh                     # 编译脚本
│  helper_ply.py                     # ply文件处理相关
│  helper_requirements.txt
│  helper_tf_util.py                  # 常用函数
│  helper_tool.py                     # 超参数设置
│  jobs_6_fold_cv_s3dis.sh
│  jobs_test_semantickitti.sh         # semantickitti测试脚本
│  LICENSE
│  main_S3DIS.py                      # S3DIS主代码
│  main_Semantic3D.py                 # Semantic3D主代码
│  main_SemanticKITTI.py              # SemanticKITTI主代码
│  RandLANet.py                       # 网络核心代码
│  README_official.md
│  tester_S3DIS.py                    # S3DIS测试
│  tester_Semantic3D.py               # Semantic3D测试
│  tester_SemanticKITTI.py            # SemanticKITTI测试
└─utils
    │  6_fold_cv.py
    │  data_prepare_s3dis.py          # SemanticKITTI数据预处理
    │  data_prepare_semantic3d.py     # semantic3d数据预处理
    │  data_prepare_semantickitti.py  # semantickitti数据预处理
    │  download_semantic3d.sh         # SemanticKITTI测试
    │  semantic-kitti.yaml
    │
    ├─cpp_wrappers
    │  │  compile_wrappers.sh
    │  │
    │  ├─cpp_subsampling
    │  │  │  setup.py
    │  │  │  wrapper.cpp
    │  │  │
    │  │  └─grid_subsampling
    │  │          grid_subsampling.cpp
    │  │          grid_subsampling.h
    │  │
    │  └─cpp_utils
    │      ├─cloud
    │      │      cloud.cpp
    │      │      cloud.h
    │      │
    │      └─nanoflann
    │              nanoflann.hpp
    │
    ├─meta
    │      anno_paths.txt
    │      class_names.txt
    │
    └─nearest_neighbors # 最近邻插值相关算法
            KDTreeTableAdaptor.h
            knn.pyx
            knn_.cxx
            knn_.h
            nanoflann.hpp
            setup.py
            test.py # 最近邻插值算法测试