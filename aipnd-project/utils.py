from enum import Enum
import json
import logging
import os
import time

import numpy as np
from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.datasets import ImageFolder
from torchvision.models.densenet import DenseNet121_Weights
from torchvision.models.resnet import ResNet18_Weights
from torchvision.models.alexnet import AlexNet_Weights
from torchvision.models.vgg import VGG16_Weights, VGG13_Weights


class DatasetType(Enum):
    TRAIN = 1
    VALIDATE = 2
    TEST = 3


def flower_data_factory(base_data_dir_path: str, dataset_type: DatasetType, batch_size=64) -> tuple[ImageFolder, DataLoader]:
    """
    Return Dataset and DataLoader corresponding to the of data set.
    """
    if dataset_type == DatasetType.TRAIN:
        data_dir_path = os.path.join(base_data_dir_path, "train")
        # Incorporate random rotation, resized crop, and horizontal flip to keep the model generalized
        data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                         transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], 
                                                              [0.229, 0.224, 0.225])])
        dataset = datasets.ImageFolder(data_dir_path, transform=data_transforms)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    elif dataset_type in (DatasetType.TEST, DatasetType.VALIDATE):
        data_dir_path = os.path.join(base_data_dir_path, "test") if dataset_type == DatasetType.TEST else os.path.join(base_data_dir_path, "valid")
        # Do not add any random transformations, only those required for processing
        data_transforms = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], 
                                                              [0.229, 0.224, 0.225])])
        dataset = datasets.ImageFolder(data_dir_path, transform=data_transforms)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    else:
        raise ValueError(f"dataset_type ({dataset_type}) not recognized")
    return dataset, data_loader


class AdaptedImageNetModel():
    def __init__(self):
        self.model = None
        self._initialized = False
        self.base_model_name = None
        self.class_label_to_class_idx = None
        self.class_label_to_class_name_file_path = None
        self.num_hidden_units = None
        self.num_output_units = None
        self.dropout = None
        self.gpu = None
        return
        
        
    def initialize(self, model_name: str, class_label_to_class_idx: dict[str, int], 
                 class_label_to_class_name_file_path: str = None, num_hidden_units: int = 512, 
                 num_output_units: int = 102, dropout: float = None, gpu: bool = False) -> None:
        """
        """
        self.base_model_name = model_name
        self.class_label_to_class_idx = class_label_to_class_idx
        self.class_label_to_class_name_file_path = class_label_to_class_name_file_path
        self.num_hidden_units = num_hidden_units
        self.num_output_units = num_output_units
        self.dropout = dropout
        self.gpu = gpu

        # set device
        if self.gpu and not torch.cuda.is_available():
                raise ValueError("gpu was set to True, but CUDA is not available")
        self.device = torch.device("cuda:0") if self.gpu else torch.device("cpu")
        
        # load pretrained model, freeze weights, replace classifier, move to device
        self._initialize_model()

        # build mapping dictionaries used in reporting results
        self._build_mapping_dictionaries()

        self._initialized = True

        return


    def initialize_from_checkpoint(self, checkpoint_file_path, gpu=False):
        """
        """
        model, checkpoint_dict = AdaptedImageNetModel.load_checkpoint(checkpoint_file_path)
        self.model = model

        # set device
        self.gpu = gpu
        if self.gpu and not torch.cuda.is_available():
                raise ValueError("gpu was set to True, but CUDA is not available")
        self.device = torch.device("cuda:0") if self.gpu else torch.device("cpu")

        # move model to device
        self.model.to(device=self.device)
        
        # check that all required keys are present
        required_keys = {"base_model_name",
                         "class_label_to_class_name_file_path", 
                         "num_input_units",
                         "num_hidden_units",
                         "num_output_units",
                         "dropout",
                         "class_label_to_class_idx",
                         "class_idx_to_class_name",
                         "class_idx_to_class_label"}
        missing_keys = required_keys - set(checkpoint_dict.keys())
        if missing_keys != set():
            raise AssertionError(f"The checkpoint dictionary is missing required keys. ({missing_keys})")

        # save required information from checkpoint_dict to the appropriate class members
        for required_key in required_keys:
            self.__dict__[required_key] = checkpoint_dict[required_key]

        self._initialized = True

        return


    def _initialize_model(self):
        """
        Load pretrained model, freeze weights, replace classifier, move to device.
        """
        # load pretrained model
        self.model, self.classifier_member_name = AdaptedImageNetModel._load_pretrained_imagenet_model(self.base_model_name)

        # freeze pretrained model parameters (so that we don't do backpropagation through them)
        for param in self.model.parameters():
            param.requires_grad = False

        # build classifier
        self.num_input_units = AdaptedImageNetModel.get_num_input_units(self.get_classifier())
        logging.debug(f"num_input_units: {self.num_input_units}")
        self.model._modules[self.classifier_member_name] = \
            AdaptedImageNetModel._build_classifier(self.num_input_units, self.num_hidden_units, self.num_output_units, self.dropout)

        # move to device
        self.model.to(device=self.device)

        return


    def get_num_input_units(classifier):
        """
        Get the number of inputs to the classifier
        """
        if type(classifier) == torch.nn.modules.container.Sequential:
            for module in classifier:
                if type(module) == torch.nn.modules.linear.Linear:
                    num_input_units = module.in_features
                    break
            else:
                raise AssertionError("Could not find the number of input units in the the classifier")
        elif type(classifier) == torch.nn.modules.linear.Linear:
            num_input_units = classifier.in_features
        
        return num_input_units

    def get_classifier(self):
        """
        Get the classification layer from the model
        """
        return self.model._modules[self.classifier_member_name]

    def _load_pretrained_imagenet_model(model_name: str):
        """
        Load pretrained model.
        """
        if model_name == "alexnet":
            model = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
            classifier_member_name = "classifier"
        elif model_name == "densenet121":
            model = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
            classifier_member_name = "classifier"
        elif model_name == "resnet18":
            model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            classifier_member_name = "fc"
        elif model_name == "vgg16":
            model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
            classifier_member_name = "classifier"
        elif model_name == "vgg13":
            model = models.vgg13(weights=VGG13_Weights.IMAGENET1K_V1)
            classifier_member_name = "classifier"
        else:
            raise ValueError(f"model_name ({model_name}) not recognized")
        return model, classifier_member_name


    def _build_classifier(num_input_units: int, num_hidden_units: int, num_output_units: int, dropout: float):
        """
        Replace the classifier in the pretrained model with a user specified neural network
        """
        module_list = [nn.Linear(num_input_units, num_hidden_units),
                    nn.ReLU()]
        if dropout is not None:
            if not (0.0 < dropout < 1.0):
                raise ValueError(f"dropout must be >0.0 and <1.0, but it was set to {dropout}")
            module_list.append(nn.Dropout(dropout))
        module_list.extend([nn.Linear(num_hidden_units, num_output_units),
                            nn.LogSoftmax(dim=1)])
        classifier = nn.Sequential(*module_list)

        return classifier
    

    def build_class_idx_to_class_label(class_label_to_class_idx: dict[str, int]) -> dict[int, str]:
        """
        Build mapping from class_idx (e.g. 0) to class_label (e.g. "1")

        Definitions:
            class_idx: integer pytorch representation of the class (i.e. the value in 
                trainset.class_to_idx)
                Ex: 0
            class_label: string pytorch class label for class derived from the image
                folder name (i.e. the key in trainset.class_to_idx)
                Ex: "1"
        Args:
            class_label_to_class_idx
                Ex: {"1" : 0, "2" : 1}
        Returns:
            class_idx_to_class_label
                Ex: {0 : "1", 1 : "2"}
        """
        class_idx_to_class_label = dict()
        for class_label, class_idx in class_label_to_class_idx.items():
            # protect against a repeated class_idx (which *should* never happen, but technically
            # could since it is the key)
            if class_idx in class_idx_to_class_label:
                raise ValueError(f"repeated class_idx ({class_idx})")
            class_idx_to_class_label[class_idx] = class_label
        return class_idx_to_class_label


    def build_class_idx_to_class_name(class_idx_to_class_label: dict[int, str], class_label_to_class_name_file_path: str) -> dict[str, str]:
        """
        Build mapping from class_idx to class_name.

        Definitions:
            class_idx: integer pytorch representation of the class (i.e. the value in 
                trainset.class_to_idx)
                Ex: 0
            class_label: string pytorch class label for class derived from the image
                folder name (i.e. the key in trainset.class_to_idx)
                Ex: "1"
            class_name: string human-readable class name from cat_to_name.json
                Ex: "pink primrose"
        Args:
            class_idx_to_class_label
                Ex: {0 : "1", 1 : "2"}
            class_label_to_class_name_file_path: file path containing information of the following
                form after load:
        Return:
            class_idx_to_class_name
                Ex: {0 : "pink primrose", 1 : "hard-leaved pocket orchid"}
        """
        with open(class_label_to_class_name_file_path, "r") as inp_file:
            class_label_to_class_name = json.load(inp_file)
        class_idx_to_class_name = dict()
        for class_idx, class_label in class_idx_to_class_label.items():
            class_name = class_label_to_class_name[class_label]
            class_idx_to_class_name[class_idx] = class_name
        return class_idx_to_class_name


    def _build_mapping_dictionaries(self):
        """
        Build and store mapping dictionaries
        """
        self.class_idx_to_class_label = AdaptedImageNetModel.build_class_idx_to_class_label(self.class_label_to_class_idx)
        self.class_idx_to_class_name = None
        if self.class_label_to_class_name_file_path is not None:
            self.class_idx_to_class_name = AdaptedImageNetModel.build_class_idx_to_class_name(self.class_idx_to_class_label, 
                                                                              self.class_label_to_class_name_file_path)
        return


    def predict(self, image_file_path, topk=5):
        """
        Predict the class (or classes) of an image using a trained deep learning model.
        """
        model_was_in_training_state = self.model.training
        self.model.eval()
        
        # reshape the 3 dim single image tensor provided by process_image to the expected
        # 4 dim tensor.  And move it to device.
        im_tensor = AdaptedImageNetModel.process_image(image_file_path)
        im_tensor = im_tensor.view(1, *im_tensor.shape)
        im_tensor = im_tensor.to(self.device)
        
        with torch.no_grad():
            logps = self.model.forward(im_tensor)
            probabilities = torch.exp(logps)
        
        topk_tensor = torch.topk(probabilities, topk, dim=1)
        predicted_topk_class_probs = topk_tensor.values.flatten().cpu().numpy()
        predicted_topk_class_idxs = topk_tensor.indices.flatten().cpu().numpy()
        

        if model_was_in_training_state:
            self.model.train()
        
        return predicted_topk_class_idxs, predicted_topk_class_probs


    def process_image(image_file_path):
        """
        Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array

        Args:
            image_file_path: file path for image to be processed
        Return:
            torch tensor of processed
        """
        with Image.open(image_file_path) as image:
            image = image.resize((255, 255))
            image = AdaptedImageNetModel._center_crop(image, 224, 224)

        # convert to a numpy array: shape = (height, width, num_channels)
        im_arr = np.array(image)

        # normalize as expected by imagenet
        norm_im_arr = AdaptedImageNetModel._normalize_for_imagenet(im_arr)

        # transpose to shape = (num_channels, height, width)
        reshaped_norm_im_arr = np.transpose(norm_im_arr, axes=(2,0,1))

        return torch.tensor(reshaped_norm_im_arr, dtype=torch.float)


    def _center_crop(image, width, height):
        """
        Crop image to a centered box with dimensions (width, height)

        Args:
            image: PIL Image to crop
            width: integer number of pixels width for cropped image
            height: integer number of pixels height for cropped image
        
        Return:
            cropped_image: cropped PIL Image
        """
        im_width, im_height = image.size
        if im_width < width or im_height < height:
            raise ValueError(f"requested cropped image size ({width}, {height}) is larger than the image ({im_width}, {im_height})")

        # determine left and right indices for crop    
        width_buffer = (im_width - width)
        left = width_buffer / 2
        right = left + width

        # determine upper and lower indices for crop    
        height_buffer = (im_height - height)
        upper = height_buffer / 2
        lower = upper + height

        # crop image
        cropped_image = image.crop((left, upper, right, lower))

        # verify that cropped image is as expected
        cropped_width, cropped_height = cropped_image.size
        if cropped_width != width or cropped_height != height:
            raise AssertionError(f"Cropped image ({cropped_width}, {cropped_height}) is not of the expected dimensions ({width}, {height})")

        return cropped_image


    def _normalize_for_imagenet(im_arr):
        """
        Perform standard normalization by channel expected by ImageNet
            norm = (scaled_im_arr - mean) / stdev
        
        Args:
            im_arr: numpy array with shape = (height, width, num_channels)
        
        Return:
            norm_im_array: normalized im_arr with shape = (height, width, num_channels)
        """
        # scale values in range of [0, 1] (im_arr values are in range [0, 255])
        scaled_im_arr = im_arr / 255.0

        # normalize
        imagenet_means = np.array([0.485, 0.456, 0.406])
        imagenet_stdevs = np.array([0.229, 0.224, 0.225])
        norm_im_arr = (scaled_im_arr - imagenet_means) / imagenet_stdevs

        return norm_im_arr


    def save_checkpoint(self, output_file_path):
        """
        Save model to checkpoint file with enough information to restore for inference or to
        continue training.
        
        This function relies on the base model being known and with only the model.classifier member
        having been modified with a NN specified using nn.Sequential.  In limited testing, it
        seems to be sufficient to use the repr for each module that is part of module.classifier.
        """
        # TODO: broader, systematic testing that this method works for the types of modules that
        # we want to test.  Explicitly verify that module types fall within "allowed" module types.
        # For this exercise, the limited manual testing is sufficient.
        # Example list:
        #       [torch.nn.Linear(in_features=1024, out_features=512, bias=True),
        #        torch.nn.ReLU(),
        #        torch.nn.Linear(in_features=512, out_features=102, bias=True),
        #        torch.nn.LogSoftmax(dim=1)]

        # throw an error if model.classifier isn't of the type expected
        classifier = self.get_classifier()
        if type(classifier) != torch.nn.modules.container.Sequential:
            raise ValueError("save_checkpoint() only works for models with model.classifier replaced with a nn.Sequential(..) definition.")
        
        # build list of repr for each module
        classifier_module_reprs = [f"torch.nn.{repr(module)}" for module in classifier]
        
        # save the classifier info, the model's state_dict, and other class members to file
        checkpoint = {"classifier_member_name" : self.classifier_member_name,
                      "classifier_module_reprs" : classifier_module_reprs,
                      "state_dict" : self.model.state_dict(),
                      "base_model_name" : self.base_model_name,
                      "class_label_to_class_name_file_path" : self.class_label_to_class_name_file_path, 
                      "num_input_units" : self.num_input_units,
                      "num_hidden_units" : self.num_hidden_units,
                      "num_output_units" : self.num_output_units,
                      "dropout" : self.dropout,
                      "class_label_to_class_idx" : self.class_label_to_class_idx,
                      "class_idx_to_class_name" : self.class_idx_to_class_name,
                      "class_idx_to_class_label" : self.class_idx_to_class_label}
        torch.save(checkpoint, output_file_path)

        return


    def load_checkpoint(checkpoint_file_path):
        """
        Load saved model from checkpoint file.
        """
        # read checkpoint file into dictionary
        checkpoint_dict = torch.load(checkpoint_file_path)

        required_keys = {"classifier_module_reprs", "state_dict", "base_model_name", "classifier_member_name", "class_label_to_class_idx"}
        missing_keys = required_keys - set(checkpoint_dict.keys())
        if missing_keys != set():
            raise AssertionError(f"The checkpoint dictionary is missing required keys. ({missing_keys})")

        # load base model
        base_model_name = checkpoint_dict["base_model_name"]
        model, classifier_member_name = AdaptedImageNetModel._load_pretrained_imagenet_model(base_model_name)
        if classifier_member_name != checkpoint_dict["classifier_member_name"]:
            raise ValueError(f"classifier_member_name ({classifier_member_name}) from loading the pretrained model does not match the value storeck in checkpoint_dict ({checkpoint_dict['classifier_member_name']})")

        # define a new classifier using the saved repr string to build up a list of objects
        classifier_modules = [eval(repr_str) for repr_str in checkpoint_dict["classifier_module_reprs"]]
        model._modules[classifier_member_name] = nn.Sequential(*classifier_modules)

        # load all tensors into model
        model.load_state_dict(checkpoint_dict["state_dict"])

        # remove the keys that are no longer necessary from checkpoint_dict before returning it
        checkpoint_dict.pop("state_dict")
        checkpoint_dict.pop("classifier_module_reprs")

        return model, checkpoint_dict


class Trainer():
    def __init__(self, model, criterion, optimizer, training_data_loader, validation_data_loader, gpu=False):
        """
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.training_data_loader = training_data_loader
        self.validation_data_loader = validation_data_loader

        # set device
        self.gpu = gpu
        if self.gpu and not torch.cuda.is_available():
                raise ValueError("gpu was set to True, but CUDA is not available")
        self.device = torch.device("cuda:0") if self.gpu else torch.device("cpu")

        # for tracking information during training
        self.total_num_epochs_trained = 0
        self.training_losses = []
        self.training_accuracies = []
        self.validation_losses = []
        self.validation_accuracies = []
        self.training_timings = []
        self.validation_testing_timings = []

        return

    def train(self, num_epochs):
        """
        Train the model
        """
        # move model tensors to device
        self.model.to(device=self.device)

        # set model to training mode
        self.model.train()

        start_time = time.time()
        for epoch in range(num_epochs):
            logging.info(f"epoch {epoch}")
            # train one epoch
            avg_training_loss, avg_training_accuracy, epoch_run_time = self._train_epoch()

            # calculate scores on validation data
            validation_loss, validation_accuracy, validation_testing_run_time = self.get_model_loss_and_accuracy()

            # save data to running lists
            self.total_num_epochs_trained += 1
            self.training_losses.append(avg_training_loss)
            self.training_accuracies.append(avg_training_accuracy)
            self.validation_losses.append(validation_loss)
            self.validation_accuracies.append(validation_accuracy)
            self.training_timings.append(epoch_run_time)
            self.validation_testing_timings.append(validation_testing_run_time)

            # print data for user
            # logging.info(f"epoch: {epoch} | run_time: {epoch_run_time:.1f} | avg training loss: {avg_training_loss} | avg training accuracy: {avg_training_accuracy} | validation loss: {validation_loss} | validation accuracy: {validation_accuracy} | validation timing: {validation_testing_run_time}")
            logging.info(f"\ttraining run time:     {epoch_run_time:.1f}")
            logging.info(f"\tvalidation run time:   {validation_testing_run_time:.1f}")
            logging.info(f"\ttraining avg loss:     {avg_training_loss:.4f}")
            logging.info(f"\ttraining avg accuracy: {avg_training_accuracy:.4f}")
            logging.info(f"\tvalidation avg loss:   {validation_loss:.4f}")
            logging.info(f"\tvalidation accuracy:   {validation_accuracy:.4f}")

        run_time = time.time() - start_time

        logging.info(f"run_time: {run_time:.1f}")

        return


    def _train_epoch(self, print_every_n_batches=30):
        """
        Train a single epoch.
        """
        losses = np.zeros(len(self.training_data_loader), dtype=np.double)
        batch_sizes = np.zeros(len(self.training_data_loader), dtype=int)
        accuracies = np.zeros(len(self.training_data_loader), dtype=np.double)
        start_time = time.time()
        for batch_num, (inputs, labels) in enumerate(self.training_data_loader):
            # print batch number periodically to show progress
            if ((batch_num + 1) % print_every_n_batches == 0) or (batch_num == len(self.training_data_loader) - 1):
                logging.info(f"\tbatch_num: {batch_num + 1} / {len(self.training_data_loader)}")
            
            # Move input and label tensors to the GPU
            inputs, labels = inputs.to(device=self.device), labels.to(device=self.device)

            # forward propagate and calculate loss
            outputs = self.model.forward(inputs)
            loss = self.criterion(outputs, labels)
            accuracy = Trainer.get_accuracy(outputs, labels)

            # backpropagate and take optimization step
            # gradients are zeroed before backpropagation since they accumulate
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # keep track of stats
            batch_size = len(labels)
            losses[batch_num] = loss.item()
            accuracies[batch_num] = accuracy
            batch_sizes[batch_num] = batch_size

        # NLL loss by default uses a "mean" reduction.  I will average over batch and not worry
        # if the last batch isn't the same size
        avg_loss = losses.sum() / len(losses)

        # Overall accuracy is most appropriately calculated by a weighted average over batch sizes.
        avg_accuracy = (accuracies * batch_sizes).sum() / batch_sizes.sum()

        run_time = time.time() - start_time

        return avg_loss, avg_accuracy, run_time


    def get_accuracy(output, labels):
        """
        Get accuracy from forward propagation output and labels.
        """
        num_correct = (torch.topk(output, 1, dim=1).indices == labels.view(-1,1)).sum().item()
        num_total = len(labels)
        accuracy = num_correct / num_total
        return accuracy


    def get_model_loss_and_accuracy(self):
        """
        Get model's loss and accuracy on the validation data set
        """
        # TODO: consider generalizing outside of just validation data loader
        start_time = time.time()
        losses = np.zeros(len(self.validation_data_loader), dtype=np.double)
        batch_sizes = np.zeros(len(self.validation_data_loader), dtype=int)
        accuracies = np.zeros(len(self.validation_data_loader), dtype=np.double)

        # set model to eval mode if hasn't been already
        self.model.eval()
        for batch_num, (inputs, labels) in enumerate(self.validation_data_loader):
            # Move input and label tensors to device
            inputs, labels = inputs.to(device=self.device), labels.to(device=self.device)

            # forward propagate (no gradient tracking)
            with torch.no_grad():
                logps = self.model.forward(inputs)

            # calculate loss and add to running total
            loss = self.criterion(logps, labels)

            # determine accuracy and add to running totals
            accuracy = Trainer.get_accuracy(logps, labels)

            batch_size = len(labels)
            losses[batch_num] = loss.item()
            accuracies[batch_num] = accuracy
            batch_sizes[batch_num] = batch_size
        
        # return model to training mode
        self.model.train()

        # calculate stats
        avg_loss = losses.sum() / len(losses)
        avg_accuracy = (accuracies * batch_sizes).sum() / batch_sizes.sum()
        run_time = time.time() - start_time

        return avg_loss, avg_accuracy, run_time


def check_training_args(args):
    """
    Verify command line arguments are valid.
    """
    # verify args contains the required parameters
    required_parameters = {"data_directory",
                           "arch",
                           "category_names",
                           "hidden_units",
                           "gpu",
                           "learning_rate",
                           "epochs",
                           "dropout",
                           "checkpoint_file_path"}
    args_set = set(args.__dict__.keys())
    missing_parameters = required_parameters - args_set
    if len(missing_parameters) != 0:
        raise ValueError(f"args is missing required parameters: {missing_parameters}")

    # verify that directories exist
    dir_paths = [args.data_directory, args.save_dir]
    for dir_path in dir_paths:
        if not os.path.isdir(dir_path):
            raise ValueError(f"Directory does not exist ({dir_path})")

    # verify files exist
    if args.category_names is not None and not os.path.isfile(args.category_names):
        raise ValueError(f"File does not exist ({args.category_names})")

    # check that learning_rate, hidden_units, epochs, and dropout values are valid
    if args.learning_rate <= 0:
        raise ValueError(f"learning_rate must be greater than 0.0, but it was set to {args.learning_rate}")
    if args.hidden_units <= 0:
        raise ValueError(f"hidden_values must be greater than zero, but it was set to {args.hidden_units}")
    if args.epochs <= 0:
        raise ValueError(f"epochs must be greater than zero, but it was set to {args.epochs}")
    if args.dropout is not None and not (0.0 < args.dropout < 1.0):
        raise ValueError(f"dropout ({args.dropout}) must be greater than zero and less than one")
    
    # if gpu flag is set, verify that the GPU is available
    if args.gpu and not torch.cuda.is_available():
        raise ValueError("GPU flag was set, but CUDA is not available!")

    return


def train(args):
    """
    Complete all steps required to train and save a model that classifies flower images into 
    one of 102 categories.

    High-level steps
        1. Load the training and validation flower data sets
        2. Load the specified ImageNet-trained base model
        3. Replace the classifier with NN with the parameters specified
        4. Train the model
        5. Save the final model to checkpoint file
    """
    # check whether training arguments are valid
    check_training_args(args)

    # Build the data loaders
    logging.info("Creating training/validation flower data loaders")
    training_dataset, training_data_loader = flower_data_factory(args.data_directory, DatasetType.TRAIN)
    validation_dataset, validation_data_loader = flower_data_factory(args.data_directory, DatasetType.VALIDATE)
    class_label_to_class_idx = training_dataset.class_to_idx
    logging.info("\ttraining_data_loader")

    # Load AdaptedImageNetModel
    logging.info(f"Loading {args.arch} base model and modifying classifier layer")
    adapted_imagenet_model = AdaptedImageNetModel()
    adapted_imagenet_model.initialize(args.arch, class_label_to_class_idx, args.category_names, 
                                      num_hidden_units=args.hidden_units, num_output_units=102, 
                                      dropout=args.dropout, gpu=args.gpu)

    # Define loss criterion and optimizer to use during training
    criterion = nn.NLLLoss()
    classifier = adapted_imagenet_model.get_classifier()
    optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)

    # Initialize the Trainer
    trainer = Trainer(adapted_imagenet_model.model, criterion, optimizer, training_data_loader, validation_data_loader, gpu=args.gpu)

    # Train
    logging.info(f"Training model for {args.epochs} epoch(s).")
    trainer.train(args.epochs)

    # Save checkpoint
    logging.info(f"Saving checkpoint file: {args.checkpoint_file_path}")
    adapted_imagenet_model.save_checkpoint(args.checkpoint_file_path)

    logging.info("Done")

    return


def check_predict_args(args):
    """
    Verify command line arguments are valid.
    """
    # verify args contains the required parameters
    required_parameters = {"image_file_path",
                           "model_file_path",
                           "top_k",
                           "category_names",
                           "gpu"}
    args_set = set(args.__dict__.keys())
    missing_parameters = required_parameters - args_set
    if len(missing_parameters) != 0:
        raise ValueError(f"args is missing required parameters: {missing_parameters}")
    
    # verify that files exist
    file_paths = [args.image_file_path, args.model_file_path]
    if args.category_names is not None:
        file_paths.append(args.category_names)
    for file_path in file_paths:
        if not os.path.isfile(file_path):
            raise ValueError(f"File path does not exist ({file_path})")
    
    # verify that top_k has been set to a valid integer
    if args.top_k <= 0:
        raise ValueError(f"top_k ({args.top_k}) must be greater than zero")

    return


def predict(args):
    """
    Complete all steps required to load a saved model and predict flower classification for
    an image.
    """
    # verify all args are present and valid
    check_predict_args(args)

    # load model from checkpoint
    logging.debug(f"Loading model")
    logging.debug(f"\tcheckpoint file: {args.model_file_path}")
    adapted_imagenet_model = AdaptedImageNetModel()
    adapted_imagenet_model.initialize_from_checkpoint(args.model_file_path, args.gpu)

    # Build the mapping dictionaries.  If args.category_names is a file, then it will create a
    # class_idx_to_class_name mapping.
    logging.debug("Building mapping dictionaries")
    adapted_imagenet_model.class_label_to_class_name_file_path = args.category_names
    if adapted_imagenet_model.class_label_to_class_name_file_path is not None:
        logging.debug(f"\tcategory names file: {adapted_imagenet_model.class_label_to_class_name_file_path}")
    adapted_imagenet_model._build_mapping_dictionaries()

    # predict probabilities of each class for image
    logging.debug(f"Predicting top {args.top_k} classification(s)")
    logging.debug(f"\timage: {args.image_file_path}")
    predicted_idxs, predicted_probs = adapted_imagenet_model.predict(args.image_file_path, topk=args.top_k)
    
    # log results
    class_idx_to_class_name = adapted_imagenet_model.class_idx_to_class_name
    class_idx_to_class_label = adapted_imagenet_model.class_idx_to_class_label
    logging.info("idx     label     prob       name")
    for class_idx, class_prob in zip(predicted_idxs, predicted_probs):
        # if json mapping file was provided, then print the class name
        if class_idx_to_class_name is not None:
            class_name = class_idx_to_class_name[class_idx]
        else:
            class_name = ""
        class_label = class_idx_to_class_label[class_idx]
        class_label_str = f"'{class_label}'"
        logging.info(f"{str(class_idx).ljust(8)}{class_label_str.ljust(10)}{class_prob:.4f}     {class_name}")