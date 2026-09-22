# include/

统一工程的头文件目录。公共工具放在 `common/`，任务专用配置/接口按任务目录组织。
这些头文件由三个可执行程序通过根目录 CMakeLists.txt 的统一 include 路径引用。
