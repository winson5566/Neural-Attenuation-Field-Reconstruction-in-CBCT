import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


columns_title  = ['MLP (HGPE)', 'SE-RAD-UNet (HashGridEncoder)', 'GT']
columns_suffix = ['baseline', 'ours', 'gt']
rows = ['Chest', 'Jaw', 'Foot', 'Abdomen']  # 已加入 Aorta

def get_filename(row, col):
    return f"{rows[row].lower()}_{columns_suffix[col]}.tiff"


fig, axes = plt.subplots(nrows=len(rows),
                         ncols=len(columns_title),
                         figsize=(9, 12),
                         constrained_layout=False)


for i in range(len(rows)):
    for j in range(len(columns_suffix)):
        ax = axes[i, j]
        fpath = os.path.join('images/', get_filename(i, j))

        if os.path.exists(fpath):
            with Image.open(fpath) as img:
                mid = img.n_frames // 2
                img.seek(mid)
                frame = img.convert('L')
                arr = np.array(frame, dtype=np.float32)
                arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
                ax.imshow(arr, cmap='gray', vmin=0, vmax=1)
        else:
            ax.text(0.5, 0.5, 'Missing', ha='center', va='center')
            ax.set_facecolor('lightgray')

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)


for ax, col in zip(axes[0], columns_title):
    ax.set_title(col, fontsize=14, pad=12)


for ax, label in zip(axes[:, 0], rows):
    ax.set_ylabel(label,
                  rotation=0,
                  ha='right',
                  va='center',
                  size=14,
                  labelpad=20)


plt.subplots_adjust(left=0.15,
                    right=0.98,
                    top=0.98,
                    bottom=0.02,
                    hspace=0.02,
                    wspace=0.02)

plt.savefig('visual_comparison.png', dpi=600)
plt.show()
