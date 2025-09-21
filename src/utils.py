import matplotlib.pyplot as plt

def plot2(img1, img2, titles=None):
    # Create a figure with 2 subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # First image (random pattern)
    axes[0].imshow(img1)

    # Second image (random pattern)
    axes[1].imshow(img2)

    if not titles is None and len(titles) == 2:
        axes[0].set_title(titles[0])
        axes[1].set_title(titles[1])

    # Remove axis ticks
    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()
