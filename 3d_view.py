import tifffile
import napari

vol = tifffile.imread('data/out/chest_50_ours_1000/1000.tiff')  # shape = (Z, H, W)

viewer = napari.Viewer()
viewer.add_image(vol, name='CT Reconstruction', colormap='gray', rendering='mip')
napari.run()