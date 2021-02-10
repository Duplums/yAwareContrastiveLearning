
PRETRAINING = 0
FINE_TUNING = 1

class Config:

    def __init__(self, mode):
        assert mode in {PRETRAINING, FINE_TUNING}, "Unknown mode: %i"%mode

        self.mode = mode

        if self.mode == PRETRAINING:
            self.batch_size = 64
            self.nb_epochs_per_saving = 1
            self.pin_mem = True
            self.num_cpu_workers = 8
            self.nb_epochs = 100
            self.cuda = True
            # Optimizer
            self.lr = 1e-4
            self.weight_decay = 5e-5
            # Hyperparameters for our y-Aware InfoNCE Loss
            self.sigma = 5 # depends on the meta-data at hand
            self.temperature = 0.1
            self.tf = "all_tf"

            # Paths to the data
            self.data_train =  "/path/to/your/training/data.npy"
            self.label_train = "/path/to/your/training/metadata.csv"

            self.data_val = "/path/to/your/training/data.npy"
            self.label_val = "/path/to/your/validation/metadata.csv"

            self.input_size = (1, 121, 145, 121)
            self.label_name = "age"

            self.checkpoint_dir = "/path/to/saving/directory/"

        elif self.mode == FINE_TUNING:
            ## We assume a classification task here
            self.batch_size = 8
            self.nb_epochs_per_saving = 10
            self.pin_mem = True
            self.num_cpu_workers = 1
            self.nb_epochs = 100
            self.cuda = True
            # Optimizer
            self.lr = 1e-4
            self.weight_decay = 5e-5

            self.pretrained_path = "/path/to/model.pth"
            self.num_classes = 2
