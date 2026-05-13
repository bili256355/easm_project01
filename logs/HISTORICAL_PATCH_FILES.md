# HISTORICAL_PATCH_FILES

本文件说明为什么整工程包里仍保留大量历史 patch/readme 文档。

---

## 保留原因
这些文件保留是为了：
- 回溯历史交付内容
- 理解旧输出的来源
- 支持后续工程/科研审计

它们包括：
- `PATCH_APPLY.txt`
- `PATCH_INSTRUCTIONS.txt`
- `PATCH_README.txt`
- `README_PATCH_APPLY.txt`
- 各层内部的 `PATCH_README*.txt`

---

## 读取规则

### 这些文件可以用来：
- 看某轮 patch 曾改过哪些文件
- 看某轮历史输出对应什么版本语义

### 这些文件不应直接用来：
- 作为当前整工程包的唯一操作说明
- 作为当前唯一推荐运行方式
- 代替根目录 README / CURRENT_PACKAGE_STATUS

---

## 当前包的推荐理解方式
- 代码与运行基线：按当前包内代码树理解
- 历史 patch：按“回溯材料”理解
- 历史 outputs：按“历史结果”理解
