
# 📦 项目中的 Git Submodule 使用指南（以 MeshCNN 为例）

> 本项目使用了 `git submodule` 来引入外部仓库 `MeshCNN`，该模块位于 `third_party/MeshCNN`。

---

## ✅ 一、克隆项目

初次克隆时请使用以下命令，确保子模块也被正确下载：

```bash
git clone https://github.com/<your-team>/vEM-Reconstruction-Geo.git
cd vEM-Reconstruction-Geo
git submodule update --init --recursive
```

---

## ✅ 二、更新子模块（例如：MeshCNN）到远程最新版本

> 如果我们更新了 `MeshCNN` 的代码并 push 了新的 commit，其他队员同步项目后也需要更新子模块：

```bash
# 确保你在主项目根目录
git submodule update --remote --merge
```

这会自动拉取远端子模块（比如 `MeshCNN`）的最新版本，并合并到当前 commit。

---

## ✅ 三、如果你要修改 MeshCNN 子模块的代码

1. 进入子模块目录：

   ```bash
   cd third_party/MeshCNN
   ```

2. 切换到你自己的分支并修改代码：

   ```bash
   git checkout -b my-feature-branch  # 或 git checkout main
   # 进行修改 ...
   git add .
   git commit -m "Add new mesh feature"
   git push origin my-feature-branch
   ```

3. 回到主项目并更新子模块的引用：

   ```bash
   cd ../..
   git add third_party/MeshCNN
   git commit -m "Update MeshCNN submodule to new commit"
   git push origin main
   ```

---

## ✅ 四、常见注意事项

| 注意事项                            | 说明                                      |
| ------------------------------- | --------------------------------------- |
| 子模块是独立 Git 仓库                   | 所以需要单独 `git add/commit/push`            |
| 每次更改子模块后必须提交主仓库                 | 因为主仓库需要记录子模块新的 commit hash              |
| `git submodule update` 不会自动切换分支 | 如果你在子模块中想切换分支，请手动 `git checkout branch` |
| `.gitmodules` 只记录路径和 URL        | 当前使用 commit 是记录在主项目的 commit 中的          |

---

## 🧪 快速测试命令（可选）

```bash
# 查看当前子模块引用的 commit
git ls-tree HEAD third_party/MeshCNN

# 查看子模块状态
git submodule status
```

---

如需帮助或出错，可以联系维护者或运行：

```bash
git submodule update --init --recursive --force
```

---

# Note

The original implementation of MeshCNN is too old. It has the following errors:
- Extremely Slow
- Incompatible with the latesest CUDA kernel
    - cause the matmul error

We may use the implementation from PyG