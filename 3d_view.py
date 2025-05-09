import tifffile
import napari

# 读入你的多页 TIFF 体
vol = tifffile.imread('data/out/chest_50_hashencoder/0030.tiff')  # shape = (Z, H, W)

# 启动 Napari 窗口
viewer = napari.Viewer()
viewer.add_image(vol, name='CT Reconstruction', colormap='gray', rendering='mip')
napari.run()