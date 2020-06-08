from Dataset import Mulitple_Training_Dataset, data_load

batch = 100

def load_data(batch_size):
    train_loader,test_loader = data_load(batch_size)
    print("Loading Data")
    print("Train \tdata num:{} (in batch size={})".format(len(train_loader),batch_size))
    print("Test \tdata num:{}".format(len(test_loader)))
    return train_loader,test_loader

if __name__ == "__main__":
    train_loader,test_loader = load_data(batch_size=batch)
    