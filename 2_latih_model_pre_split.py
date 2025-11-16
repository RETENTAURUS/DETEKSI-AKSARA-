import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import cv2
import math

# --- Variabel Konfigurasi ---
IMG_HEIGHT = 64
IMG_WIDTH = 256
BATCH_SIZE = 32
EPOCHS = 50   # early stopping akan menghentikan otomatis
EARLY_STOPPING_PATIENCE = 10
DATASET_PATH = 'karakter3'

# --- 1. Fungsi Memuat Data ---
def load_dataset_from_folder(dataset_path):
    data = []
    for class_name in os.listdir(dataset_path):
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_dir):
            continue
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            data.append((img_path, class_name))
    return data

# --- 2. Preprocessing dengan CLAHE ---
def preprocess_image(image_path, img_height, img_width):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("File tidak bisa dibaca")
        img = cv2.resize(img, (img_width, img_height))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img = clahe.apply(img)
        img = img.astype(np.float32)/255.0
        img = 1.0 - img
        img = np.expand_dims(img, axis=-1)
        return img
    except Exception as e:
        print(f"Error memproses gambar {image_path}: {e}")
        return None

def vectorize_label(label, class_to_num_map):
    try:
        return tf.convert_to_tensor(class_to_num_map[label], dtype=tf.int32)
    except Exception as e:
        print(f"Error vektorisasi label {label}: {e}")
        return None

# --- 3. Membuat tf.data.Dataset dengan augmentasi aman ---
def create_tf_dataset(data_pairs, class_to_num_map, batch_size, augment=False):
    image_paths = [item[0] for item in data_pairs]
    labels = [item[1] for item in data_pairs]

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    def map_fn(path, lbl):
        image, label_vec = tf.py_function(
            func=lambda p, l: (
                preprocess_image(p.numpy().decode('utf-8'), IMG_HEIGHT, IMG_WIDTH),
                vectorize_label(l.numpy().decode('utf-8'), class_to_num_map)
            ),
            inp=[path, lbl],
            Tout=[tf.float32, tf.int32]
        )
        image.set_shape([IMG_HEIGHT, IMG_WIDTH, 1])
        label_vec.set_shape([])
        return image, label_vec

    dataset = dataset.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.filter(lambda img, lbl: img is not None and lbl is not None)

    if augment:
        def augment_fn(img, lbl):
            # Rotasi kecil ±10 derajat
            img = tf.image.rot90(img, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
            # Zoom kecil
            img = tf.image.resize(img, [IMG_HEIGHT+4, IMG_WIDTH+8])
            img = tf.image.random_crop(img, size=[IMG_HEIGHT, IMG_WIDTH, 1])
            # Brightness & contrast jitter
            img = tf.image.random_brightness(img, max_delta=0.2)
            img = tf.image.random_contrast(img, 0.8, 1.2)
            # Gaussian noise ringan
            noise = tf.random.normal(shape=tf.shape(img), mean=0.0, stddev=0.02, dtype=tf.float32)
            img = tf.clip_by_value(img + noise, 0.0, 1.0)
            return img, lbl
        dataset = dataset.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# --- 4. Model CNN dengan regularisasi ---
def build_cnn_model(img_height, img_width, num_classes):
    input_img = layers.Input(shape=(img_height, img_width,1), name="image")
    x = layers.Conv2D(32,(3,3),activation="relu",padding="same",
                      kernel_regularizer=keras.regularizers.l2(0.001))(input_img)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(64,(3,3),activation="relu",padding="same",
                      kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=input_img, outputs=output)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# --- 5. Main ---
def main():
    data = load_dataset_from_folder(DATASET_PATH)
    if not data:
        print("Error: Dataset kosong.")
        return

    np.random.shuffle(data)
    n = len(data)
    train_data = data[:int(0.7*n)]
    valid_data = data[int(0.7*n):int(0.9*n)]
    test_data  = data[int(0.9*n):]

    all_labels = [label for _, label in data]
    class_names = sorted(list(set(all_labels)))
    class_to_num_map = {cls: i for i, cls in enumerate(class_names)}
    print(f"Kelas ditemukan: {class_names}")

    train_dataset = create_tf_dataset(train_data, class_to_num_map, BATCH_SIZE, augment=True).repeat()
    valid_dataset = create_tf_dataset(valid_data, class_to_num_map, BATCH_SIZE).repeat()
    test_dataset  = create_tf_dataset(test_data, class_to_num_map, BATCH_SIZE)

    steps_per_epoch = math.ceil(len(train_data)/BATCH_SIZE)
    validation_steps = math.ceil(len(valid_data)/BATCH_SIZE)

    num_classes = len(class_names)
    model = build_cnn_model(IMG_HEIGHT, IMG_WIDTH, num_classes)
    model.summary()

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True)
    checkpoint = keras.callbacks.ModelCheckpoint("model_cnn_terbaik.keras", save_best_only=True)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1)

    history = model.fit(train_dataset,
                        validation_data=valid_dataset,
                        epochs=EPOCHS,
                        callbacks=[early_stopping, checkpoint, reduce_lr],
                        steps_per_epoch=steps_per_epoch,
                        validation_steps=validation_steps)

    print("Training selesai.")

    best_model = keras.models.load_model("model_cnn_terbaik.keras")
    test_loss, test_acc = best_model.evaluate(test_dataset)
    print(f"Final Test Loss: {test_loss:.4f}, Akurasi: {test_acc*100:.2f}%")

if _name=="main_":
    main()