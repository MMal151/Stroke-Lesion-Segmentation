import matplotlib.pyplot as plt


def show_history(history, validation: bool = False):
    if validation:
        # Loss
        fig, axes = plt.subplots(figsize=(20, 5))
        # Train
        axes.plot(history.epoch, history.history['loss'], color='r', label='Train')
        axes.plot(history.epoch, history.history['val_loss'], color='b', label='Val')
        axes.set_xlabel('Epoch')
        axes.set_ylabel('Loss')
        axes.legend()
        plt.savefig('loss.jpg')
        plt.show()
    else:
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        # loss
        axes[0].plot(history.epoch, history.history['loss'])
        axes[0].set_title('Train')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')


idx = 0


def show_data_points(img, label):
    global idx
    idx = idx + 1
    mid_slice = 120
    print(f"Image: {img[120]}")
    plt.imshow(img[mid_slice], cmap='bone')
    plt.axis('off')
    plt.savefig(f"img_{idx}.jpg")
    print(f"Image: {label[120]}")
    plt.imshow(label[mid_slice], cmap='bone')
    plt.axis('off')
    plt.savefig(f"lbl_{idx}.jpg")
