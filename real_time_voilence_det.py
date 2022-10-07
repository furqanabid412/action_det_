from dataload import DataSET
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
import matplotlib.pyplot as plt
import wandb
from keras.utils import to_categorical

from keras.applications.mobilenet_v2 import MobileNetV2
from sklearn.model_selection import train_test_split
from keras.layers import *
from keras.models import Sequential
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.utils import plot_model
import tensorflow as tf
from collections import deque
import cv2
import random

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract = True, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

def pytorch_training(train_loader,test_loader,device):

    model,input_size =initialize_model(model_name, num_classes,feature_extract,True)
    # print(model_ft)

    model = model.to(device)
    params_to_update = model.parameters()

    if feature_extract:
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)

    optimizer = optim.Adam(params_to_update, lr=0.001)
    criterion = nn.CrossEntropyLoss()


    since = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                data_loader = train_loader
            else:
                model.eval()
                data_loader = test_loader

            avg_loss = AverageMeter()
            avg_accu = AverageMeter()

            # Iterate over data.
            for ite,(inputs, labels) in enumerate(data_loader):
                for x in range(batch_size):
                    img = inputs[x,:,:,:].numpy()
                    img = np.rollaxis(img,1)
                    img = np.rollaxis(img, 2,1)
                    cv2.imshow('image', img)
                    # cv2.resizeWindow('image', 700, 700)
                    cv2.waitKey()

                inputs = inputs.float().to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    pred=torch.argmax(outputs,dim=1)
                    one_hot_pred = to_categorical(pred.cpu())
                    one_hot_pred =torch.from_numpy(one_hot_pred).cuda()
                    # _, preds = torch.max(outputs, 1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    mini_batch_accu = torch.sum(one_hot_pred == labels.data) / (2*batch_size)
                avg_loss.update(loss.item(),batch_size)
                avg_accu.update(mini_batch_accu,batch_size)
                wandb.log({'Loss':loss.item(),'Acc':mini_batch_accu,'Avg_Loss':avg_loss.avg,'Avg_Accu':avg_accu.avg})
                # print('accu',mini_batch_accu.item(),'avg_accu',avg_accu.avg.item(),'sum',torch.sum(one_hot_pred == labels.data).item())
                # print('{} {} of {} Loss: {:.8f} Acc: {:.8f} Avg_Loss: {:.8f} Avg_Acc: {:.8f}'.format
                #       (phase, ite, len(data_loader), loss.item(), mini_batch_accu,avg_loss.avg,avg_accu.avg))

            wandb.log({'Epoch Loss':avg_loss.avg, 'Epoch Acc': avg_accu.avg})
            # deep copy the model
            if phase == 'val' and avg_accu.avg > best_acc:
                best_acc = avg_accu.avg
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(avg_accu.avg )


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    wandb.finish()


def mobilenetV2():
    SEQUENCE_LENGTH,IMAGE_HEIGHT,IMAGE_WIDTH = 40,64,64
    CLASSES_LIST = ['NonViolence', 'Violence']

    mobilenet = MobileNetV2(include_top=False, weights="imagenet")
    # Fine-Tuning to make the last 40 layer trainable
    mobilenet.trainable = True

    print(len(mobilenet.layers))
    for layer in mobilenet.layers[:-40]:
        # print('name',layer.name)
        layer.trainable = False

    model = Sequential()
    # Specifying Input to match features shape
    model.add(Input(shape=(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
    # Passing mobilenet in the TimeDistributed layer to handle the sequence
    model.add(TimeDistributed(mobilenet))
    model.add(Dropout(0.25))
    model.add(TimeDistributed(Flatten()))
    lstm_fw = LSTM(units=32)
    lstm_bw = LSTM(units=32, go_backwards=True)
    model.add(Bidirectional(lstm_fw, backward_layer=lstm_bw))
    model.add(Dropout(0.25))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(len(CLASSES_LIST), activation='softmax'))
    model.summary()
    return model


def plot_metric(model_training_history, metric_name_1, metric_name_2, plot_name):
    metric_value_1 = model_training_history.history[metric_name_1]
    metric_value_2 = model_training_history.history[metric_name_2]

    # Get the Epochs Count
    epochs = range(len(metric_value_1))

    plt.plot(epochs, metric_value_1, 'blue', label=metric_name_1)
    plt.plot(epochs, metric_value_2, 'orange', label=metric_name_2)

    plt.title(str(plot_name))

    plt.legend()


# To show Random Frames from the saved output predicted video (output predicted video doesn't show on the notebook but can be downloaded)
def show_pred_frames(pred_video_path,ax):
    plt.figure(figsize=(20, 15))

    video_reader = cv2.VideoCapture(pred_video_path)

    # Get the number of frames in the video.
    frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get Random Frames from the video then Sort it
    random_range = sorted(random.sample(range(SEQUENCE_LENGTH, frames_count), 12))

    for counter, random_index in enumerate(random_range, 1):

        plt.subplot(5, 4, counter)

        # Set the current frame position of the video.
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, random_index)

        ok, frame = video_reader.read()

        if not ok:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        plt.imshow(frame);
        ax.figure.set_size_inches(20, 20);
        plt.tight_layout()

    video_reader.release()


def predict_frames(video_file_path, output_file_path, SEQUENCE_LENGTH,MoBiLSTM_model):
    SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH = 40, 64, 64
    CLASSES_LIST = ['NonViolence', 'Violence']

    # Read from the video file.
    video_reader = cv2.VideoCapture(video_file_path)

    # Get the width and height of the video.
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # VideoWriter to store the output video in the disk.
    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                                   video_reader.get(cv2.CAP_PROP_FPS), (original_video_width, original_video_height))

    # Declare a queue to store video frames.
    frames_queue = deque(maxlen=SEQUENCE_LENGTH)

    # Store the predicted class in the video.
    predicted_class_name = ''

    # Iterate until the video is accessed successfully.
    while video_reader.isOpened():

        ok, frame = video_reader.read()

        if not ok:
            break

        # Resize the Frame to fixed Dimensions.
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))

        # Normalize the resized frame
        normalized_frame = resized_frame / 255

        # Appending the pre-processed frame into the frames list.
        frames_queue.append(normalized_frame)

        # We Need at Least number of SEQUENCE_LENGTH Frames to perform a prediction.
        # Check if the number of frames in the queue are equal to the fixed sequence length.
        if len(frames_queue) == SEQUENCE_LENGTH:
            # Pass the normalized frames to the model and get the predicted probabilities.
            predicted_labels_probabilities = MoBiLSTM_model.predict(np.expand_dims(frames_queue, axis=0))[0]

            # Get the index of class with highest probability.
            predicted_label = np.argmax(predicted_labels_probabilities)

            # Get the class name using the retrieved index.
            predicted_class_name = CLASSES_LIST[predicted_label]

        # Write predicted class name on top of the frame.
        if predicted_class_name == "Violence":
            cv2.putText(frame, predicted_class_name, (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 12)
        else:
            cv2.putText(frame, predicted_class_name, (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 12)

        # Write The frame into the disk using the VideoWriter
        video_writer.write(frame)

    video_reader.release()
    video_writer.release()

def tf_train(features_train,labels_train):
    MoBiLSTM_model = mobilenetV2()
    # plot_model(MoBiLSTM_model, to_file='MobBiLSTM_model_structure_plot.png', show_shapes=True, show_layer_names=True)

    # Create Early Stopping Callback to monitor the accuracy
    early_stopping_callback = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

    # Create ReduceLROnPlateau Callback to reduce overfitting by decreasing learning

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.6,patience=5,min_lr=0.00005,verbose=1)
    # Compiling the model
    MoBiLSTM_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=["accuracy"])
    # Fitting the model
    MobBiLSTM_model_history = MoBiLSTM_model.fit(x=features_train, y=labels_train, epochs=50, batch_size=8,
                                                 shuffle=True, validation_split=0.2,
                                                 callbacks=[early_stopping_callback, reduce_lr])

    model_evaluation_history = MoBiLSTM_model.evaluate(features_test, labels_test)
    plot_metric(MobBiLSTM_model_history, 'loss', 'val_loss', 'Total Loss vs Total Validation Loss')
    plot_metric(MobBiLSTM_model_history, 'accuracy', 'val_accuracy', 'Total Loss vs Total Validation Loss')

    # labels_predict = MoBiLSTM_model.predict(features_test)
    # # %%
    # # Decoding the data to use in Metrics
    # labels_predict = np.argmax(labels_predict, axis=1)
    # labels_test_normal = np.argmax(labels_test, axis=1)
    # from sklearn.metrics import accuracy_score
    # AccScore = accuracy_score(labels_predict, labels_test_normal)
    # print('Accuracy Score is : ', AccScore)
    #
    # import seaborn as sns
    # from sklearn.metrics import confusion_matrix
    #
    # ax = plt.subplot()
    # cm = confusion_matrix(labels_test_normal, labels_predict)
    # sns.heatmap(cm, annot=True, fmt='g', ax=ax)
    #
    # ax.set_xlabel('Predicted labels')
    # ax.set_ylabel('True labels')
    # ax.set_title('Confusion Matrix')
    # ax.xaxis.set_ticklabels(['True', 'False'])
    # ax.yaxis.set_ticklabels(['NonViolence', 'Violence'])
    #
    # from sklearn.metrics import classification_report
    #
    # ClassificationReport = classification_report(labels_test_normal, labels_predict)
    # print('Classification Report is : \n', ClassificationReport)
    # plt.style.use("default")


if __name__ == '__main__':
    # wandb.init(project='voilence_detection_binary')

    train_dataset,test_dataset = DataSET(trainset=True),DataSET(trainset=False)

    features_train, labels_train = train_dataset.get_data()
    features_test, labels_test = test_dataset.get_data()

    tf_train(features_train, labels_train)

    # for 2D CNN part
    # num_classes = 2
    # batch_size = 16
    # feature_extract = False
    # model_name = 'resnet'
    # num_epochs = 100
    # is_inception = False

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # # DATALOADER (pytorch)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size,shuffle=True, num_workers=0, pin_memory=True,drop_last=True)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size,shuffle=True, num_workers=0, pin_memory=True,drop_last=True)


    print('end')


