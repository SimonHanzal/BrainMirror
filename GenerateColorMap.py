import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

x,y,c = zip(*np.random.rand(30,3)*4-2)

norm=plt.Normalize(-2,2)
# cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["tan", "paleturquoise"])
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["darkviolet", "green"])
rgb_values = np.round(cmap(np.linspace(0, 1, 256)) * 256,0)
rgb_values = rgb_values[:,0:3]


save = []
for i in range(256):
    rgba = cmap(i)
    # rgb2hex accepts rgb or rgba
    save = np.append(save, matplotlib.colors.rgb2hex(rgba))
np.save("data//colour_maps.npy", save, allow_pickle=True)
pass


plt.scatter(x,y,c=c, cmap=cmap, norm=norm)
plt.colorbar()
plt.show()