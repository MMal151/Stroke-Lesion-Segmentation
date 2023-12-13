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
