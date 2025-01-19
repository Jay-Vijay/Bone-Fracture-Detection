import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

def load_path(path, part):
    """
    Load X-ray dataset from local directories.
    """
    dataset = []
    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        if os.path.isdir(folder_path):
            for body in os.listdir(folder_path):
                if body == part:
                    body_part = body
                    path_p = os.path.join(folder_path, body)
                    for id_p in os.listdir(path_p):
                        patient_id = id_p
                        path_id = os.path.join(path_p, id_p)
                        for lab in os.listdir(path_id):
                            if lab.split('_')[-1] == 'positive':
                                label = 'fractured'
                            elif lab.split('_')[-1] == 'negative':
                                label = 'normal'
                            path_l = os.path.join(path_id, lab)
                            for img in os.listdir(path_l):
                                img_path = os.path.join(path_l, img)
                                dataset.append({
                                    'body_part': body_part,
                                    'patient_id': patient_id,
                                    'label': label,
                                    'image_path': img_path
                                })
    return dataset


def trainPart(part):
    """
    Train the model for a specific body part.
    """
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(THIS_FOLDER, 'Dataset')
    
    # Load dataset
    data = load_path(image_dir, part)
    labels = []
    filepaths = []

    # Prepare labels and filepaths
    for row in data:
        labels.append(row['label'])
        filepaths.append(row['image_path'])

    filepaths = pd.Series(filepaths, name='Filepath').astype(str)
    labels = pd.Series(labels, name='Label')

    images = pd.concat([filepaths, labels], axis=1)

    # Split dataset into train, test (10%) and then train split into validation (20%)
    train_df, test_df = train_test_split(images, train_size=0.9, shuffle=True, random_state=1)

    # Image data generators with preprocessing and augmentation
    train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        horizontal_flip=True,
        preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
        validation_split=0.2
    )

    test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.resnet50.preprocess_input
    )

    train_images = train_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=64,
        shuffle=True,
        seed=42,
        subset='training'
    )

    val_images = train_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=64,
        shuffle=True,
        seed=42,
        subset='validation'
    )

    test_images = test_generator.flow_from_dataframe(
        dataframe=test_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=False
    )

    # Load ResNet50 as the base model
    pretrained_model = tf.keras.applications.resnet50.ResNet50(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )

    # Freeze the pretrained model for faster training
    pretrained_model.trainable = False

    # Add custom layers on top of ResNet50
    inputs = pretrained_model.input
    x = tf.keras.layers.Dense(128, activation='relu')(pretrained_model.output)
    x = tf.keras.layers.Dense(50, activation='relu')(x)
    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    print(f"-------Training {part}-------")

    # Callbacks to stop early if the model starts overfitting
    callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    # Train the model
    history = model.fit(
        train_images,
        validation_data=val_images,
        epochs=25,
        callbacks=[callbacks]
    )

    # Save the trained model
    model.save(os.path.join(THIS_FOLDER, f"weights/ResNet50_{part}_frac.h5"))

    # Evaluate the model on the test set
    results = model.evaluate(test_images, verbose=0)
    print(f"{part} Results:")
    print(results)
    print(f"Test Accuracy: {np.round(results[1] * 100, 2)}%")

    # Plot and save accuracy and loss graphs
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    figAcc = plt.gcf()
    my_file = os.path.join(THIS_FOLDER, f"plots/FractureDetection/{part}_Accuracy.jpeg")
    figAcc.savefig(my_file)
    plt.clf()

    # Plot and save loss graphs
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    figLoss = plt.gcf()
    my_file = os.path.join(THIS_FOLDER, f"plots/FractureDetection/{part}_Loss.jpeg")
    figLoss.savefig(my_file)
    plt.clf()


# Run the function for each body part category
categories_parts = ["Elbow", "Hand", "Shoulder"]
for category in categories_parts:
    trainPart(category)
