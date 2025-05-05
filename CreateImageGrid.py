import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# —— 1. 列 + 行（记得加上 Aorta）
columns = ['GT', 'Baseline', 'Ours']
rows    = ['Chest', 'Jaw', 'Foot', 'Abdomen',]

def get_filename(row, col):
    return f"{rows[row].lower()}_{columns[col].lower().replace(' ', '')}.tiff"

# —— 2. 创建 5×3 的子图
fig, axes = plt.subplots(nrows=len(rows),
                         ncols=len(columns),
                         figsize=(9, 12),
                         constrained_layout=False)

# —— 3. 逐个子图填图
for i in range(len(rows)):
    for j in range(len(columns)):
        ax = axes[i, j]
        fpath = os.path.join('images/', get_filename(i, j))

        if os.path.exists(fpath):
            with Image.open(fpath) as img:
                mid = img.n_frames // 2
                img.seek(mid)
                frame = img.convert('L')
                arr = np.array(frame, dtype=np.float32)
                arr = (arr - arr.min())/(arr.max() - arr.min() + 1e-8)
                ax.imshow(arr, cmap='gray', vmin=0, vmax=1)
        else:
            ax.text(0.5, 0.5, 'Missing', ha='center', va='center')
            ax.set_facecolor('lightgray')

        # —— 4. 隐藏刻度和边框，但不要用 axis('off')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

# —— 5. 设置列标题
for ax, col in zip(axes[0], columns):
    ax.set_title(col, fontsize=14, pad=12)

# —— 6. 设置行标题（靠子图左侧）
for ax, label in zip(axes[:, 0], rows):
    ax.set_ylabel(label,
                  rotation=0,
                  ha='right',
                  va='center',
                  size=14,
                  labelpad=20)

# —— 7. 调整左右边距，防止标签被裁剪
plt.subplots_adjust(left=0.15,
                    right=0.98,
                    top=0.98,
                    bottom=0.02,
                    hspace=0.02,
                    wspace=0.02)

plt.savefig('visual_comparison.tiff', dpi=300)
plt.show()
