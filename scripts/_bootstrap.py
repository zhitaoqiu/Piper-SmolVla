"""CLI 脚本路径初始化。

允许从仓库根目录直接运行 `python scripts/xxx.py`，无需手动设置
`PYTHONPATH=src`。这里只改 Python import 路径，不连接硬件、不发送动作。
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
