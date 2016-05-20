class Params:

    architecture = ""

    image_size = ""
    batch_size = ""
    featuremap_size_all = ""
    filter_size_all = ""
    dropout_ratio = ""
    n_epochs = ""
    randomize_images = ""
    num_of_train_iterator = ""
    num_of_val_iterator = ""
    num_train_samples = ""
    num_val_samples = ""
    n_classes = 3

    # parameters for Adam gradient update
    learning_rate_schedule_adam = ""
    beta1= ""
    beta2= ""
    epsilon= ""
    l2_Lambda = ""
    epoch_tolerance= ""
    accuracy_tolerance = ""
    lr_decay = ""
    L2_decay = ""
    DropOut_decay = ""

    def __init__(self, path=None):
        if path is not None:
            self.load_parameters(path)

    def set(self, parameters_Dictionary):

        self.architecture = parameters_Dictionary['architecture'][0]

        self.image_size = int(parameters_Dictionary['image_size'][0])
        self.batch_size = int(parameters_Dictionary['batch_size'][0])
        self.featuremap_size_all = map(int, parameters_Dictionary['featuremap_size_all'])
        self.filter_size_all = map(int, parameters_Dictionary['filter_size_all'])
        self.dropout_ratio = float(parameters_Dictionary['dropout_ratio'][0])
        self.n_epochs = int(parameters_Dictionary['n_epochs'][0])
        self.randomize_images = int(parameters_Dictionary['randomize_images'][0])
        self.num_of_train_iterator = map(int,parameters_Dictionary['num_of_train_iterator'])
        self.num_of_val_iterator = map(int,parameters_Dictionary['num_of_val_iterator'])
        self.num_train_samples = map(int, parameters_Dictionary['num_train_samples'])
        self.num_val_samples = map(int, parameters_Dictionary['num_val_samples'])
        self.data_level = int(parameters_Dictionary['data_level'][0])
        self.masks_level = int(parameters_Dictionary['masks_level'][0])
        # parameters for Adam gradient update
        self.learning_rate_schedule_adam =  map(float, parameters_Dictionary['learning_rate_schedule_adam'])
        self.beta1 = float(parameters_Dictionary['beta1'][0])
        self.beta2 = float(parameters_Dictionary['beta2'][0])
        self.epsilon = float(parameters_Dictionary['epsilon'][0])
        self.l2_Lambda = float(parameters_Dictionary['l2_Lambda'][0])
        self.epoch_tolerance = int(parameters_Dictionary['epoch_tolerance'][0])
        self.accuracy_tolerance = float(parameters_Dictionary['accuracy_tolerance'][0])
        self.lr_decay = float(parameters_Dictionary['lr_decay'][0])
        self.L2_decay = float(parameters_Dictionary['L2_decay'][0])
        self.DropOut_decay = float(parameters_Dictionary['DropOut_decay'][0])



    def load_parameters(self, path):
        with open(path) as f:
                network_parameters = {}
                for line in f:

                    # Comments in params file
                    if line.strip()[0] == "#":
                        continue

                    [var, val] = line.translate(None," ")[:-1].split("=")
                    network_parameters[str(var)] = list(val.split(','))

        self.set(network_parameters)
        self.dictionary = network_parameters
