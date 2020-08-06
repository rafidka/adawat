from matplotlib import pyplot as plt


def plot_training_progress(losses, train_accuracies, test_accuracies):
    plt.figure(num=None, figsize=(8, 6))
    plt.plot(losses)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Mean loss', fontsize=14)

    plt.figure(num=None, figsize=(8, 6))
    plt.plot(range(len(train_accuracies)), train_accuracies,
             range(len(test_accuracies)), test_accuracies)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.legend(['Train accuracy', 'test accuracy'])
    plt.ylim([0, 1])
