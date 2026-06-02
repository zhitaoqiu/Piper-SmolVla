"""本仓库的 Python 启动补丁。

系统 ROS 2 环境会向当前 conda env 注入 pytest 插件入口，其中部分插件依赖
不同 Python 版本的包，导致本项目测试在收集阶段失败。禁用 pytest 第三方
插件自动加载，保留项目自身测试的可重复性。
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
