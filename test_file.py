from tumorset import TumorSet
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

test_data = TumorSet('./data/test')
test_dataloader = DataLoader(test_data, batch_size=4, shuffle=True)

test_features, test_segmentation = next(iter(test_dataloader))
print(f"Feature batch shape: {test_features.size()}")
print(f"Labels batch shape: {test_segmentation.size()}")
img = test_features[0].squeeze()
label = test_segmentation[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")

