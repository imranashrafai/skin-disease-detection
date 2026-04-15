import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0, ConvNeXtTiny
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from collections import Counter
import os
BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)
SEED = 42
EPOCHS = 20
LEARNING_RATE = 1e-3
tf.random.set_seed(SEED)
np.random.seed(SEED)
DATA_DIR = '/kaggle/input/skin-diseases-image-dataset/IMG_CLASSES'
all_images = []
all_labels = []
class_names = sorted(os.listdir(DATA_DIR))
class_to_idx = {name: idx for idx, name in enumerate(class_names)}

print(f"\nScanning directory: {DATA_DIR}")
print(f"Found {len(class_names)} classes")
print(f"Classes: {class_names}")

for class_name in class_names:
    class_dir = os.path.join(DATA_DIR, class_name)
    if not os.path.isdir(class_dir):
        continue
    
    for img_name in os.listdir(class_dir):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            img_path = os.path.join(class_dir, img_name)
            all_images.append(img_path)
            all_labels.append(class_to_idx[class_name])

all_images = np.array(all_images)
all_labels = np.array(all_labels)

print(f"\nTotal images found: {len(all_images)}")
print(f"Number of classes: {len(class_names)}")
class_counts = Counter(all_labels)
for idx, class_name in enumerate(class_names):
    print(f"  {class_name}: {class_counts[idx]} images")

train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    all_images, 
    all_labels,
    test_size=0.2,
    stratify=all_labels,
    random_state=SEED
)

val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths,
    temp_labels,
    test_size=0.5,
    stratify=temp_labels,
    random_state=SEED
)

print(f"\nSplit Summary:")
print(f"  Train: {len(train_paths)} images ({len(train_paths)/len(all_images)*100:.1f}%)")
print(f"  Validation: {len(val_paths)} images ({len(val_paths)/len(all_images)*100:.1f}%)")
print(f"  Test: {len(test_paths)} images ({len(test_paths)/len(all_images)*100:.1f}%)")

def plot_class_distribution(train_labels, val_labels, test_labels, class_names):
    """Plot class distribution across splits"""
    
    train_counts = Counter(train_labels)
    val_counts = Counter(val_labels)
    test_counts = Counter(test_labels)
    
    # Prepare data for plotting
    x = np.arange(len(class_names))
    width = 0.25
    
    train_vals = [train_counts[i] for i in range(len(class_names))]
    val_vals = [val_counts[i] for i in range(len(class_names))]
    test_vals = [test_counts[i] for i in range(len(class_names))]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.bar(x - width, train_vals, width, label='Train', alpha=0.8, color='#2E86AB')
    ax.bar(x, val_vals, width, label='Validation', alpha=0.8, color='#A23B72')
    ax.bar(x + width, test_vals, width, label='Test', alpha=0.8, color='#F18F01')
    
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
    ax.set_title('Class Distribution Across Train/Val/Test Sets', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n✓ Class distribution plotted successfully!")

plot_class_distribution(train_labels, val_labels, test_labels, class_names)

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomTranslation(0.1, 0.1),
], name="data_augmentation")

def create_dataset(image_paths, labels, preprocess_fn, augment=False, shuffle=True):
    """Create TF dataset from paths and labels"""
    
    def load_and_preprocess(path, label):
        # Read image
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        return img, label
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000, seed=SEED)
    
    dataset = dataset.batch(BATCH_SIZE)
    
    # Apply augmentation
    if augment:
        dataset = dataset.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    
    # Apply preprocessing
    dataset = dataset.map(
        lambda x, y: (preprocess_fn(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    return dataset.prefetch(tf.data.AUTOTUNE)

def preprocess_efficientnet(x):
    return tf.keras.applications.efficientnet.preprocess_input(x)

train_ds_eff = create_dataset(train_paths, train_labels, preprocess_efficientnet, augment=True, shuffle=True)
val_ds_eff = create_dataset(val_paths, val_labels, preprocess_efficientnet, augment=False, shuffle=False)
test_ds_eff = create_dataset(test_paths, test_labels, preprocess_efficientnet, augment=False, shuffle=False)

def build_efficientnet_model(num_classes):    
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=INPUT_SHAPE,
        pooling='avg'
    )
    
    inputs = base_model.input
    x = base_model.output
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="EfficientNetB0_SkinDisease")
    
    return model
num_classes = len(class_names)
model_eff = build_efficientnet_model(num_classes)
model_eff.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_eff = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
    keras.callbacks.ModelCheckpoint('best_efficientnet_skin.keras', monitor='val_accuracy', save_best_only=True, verbose=1)
]
history_eff = model_eff.fit(
    train_ds_eff,
    validation_data=val_ds_eff,
    epochs=EPOCHS,
    callbacks=callbacks_eff,
    verbose=1
)
def plot_training_history(history, model_name):
    """Plot training and validation metrics"""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2, color='#2E86AB')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2, color='#A23B72')
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    axes[0].set_title(f'{model_name} - Accuracy', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2, color='#2E86AB')
    axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2, color='#A23B72')
    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[1].set_title(f'{model_name} - Loss', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
plot_training_history(history_eff, "EfficientNetB0")
test_loss_eff, test_accuracy_eff = model_eff.evaluate(test_ds_eff, verbose=1)

print(f"\n✓ EfficientNetB0 Test Results:")
print(f"  - Test Accuracy: {test_accuracy_eff*100:.2f}%")
print(f"  - Test Loss: {test_loss_eff:.4f}")

y_pred_eff = []
y_true_eff = []

for images, labels in test_ds_eff:
    preds = model_eff.predict(images, verbose=0)
    y_pred_eff.extend(np.argmax(preds, axis=1))
    y_true_eff.extend(labels.numpy())

y_pred_eff = np.array(y_pred_eff)
y_true_eff = np.array(y_true_eff)

from sklearn.metrics import confusion_matrix, classification_report

cm_eff = confusion_matrix(y_true_eff, y_pred_eff)

plt.figure(figsize=(12, 10))
sns.heatmap(cm_eff, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Count'})
plt.title('EfficientNetB0 - Confusion Matrix', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=12, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

def preprocess_convnext(x):
    return tf.keras.applications.convnext.preprocess_input(x)

train_ds_cnx = create_dataset(train_paths, train_labels, preprocess_convnext, augment=True, shuffle=True)
val_ds_cnx = create_dataset(val_paths, val_labels, preprocess_convnext, augment=False, shuffle=False)
test_ds_cnx = create_dataset(test_paths, test_labels, preprocess_convnext, augment=False, shuffle=False)
def build_convnext_model(num_classes):
    
    
    base_model = ConvNeXtTiny(
        include_top=False,
        weights='imagenet',
        input_shape=INPUT_SHAPE,
        pooling='avg'
    )
    
    inputs = base_model.input
    x = base_model.output
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="ConvNeXtTiny_SkinDisease")
    
    return model

model_cnx = build_convnext_model(num_classes)


model_cnx.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_cnx = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
    keras.callbacks.ModelCheckpoint('best_convnext_skin.keras', monitor='val_accuracy', save_best_only=True, verbose=1)
]

history_cnx = model_cnx.fit(
    train_ds_cnx,
    validation_data=val_ds_cnx,
    epochs=EPOCHS,
    callbacks=callbacks_cnx,
    verbose=1
)

plot_training_history(history_cnx, "ConvNeXtTiny")
test_loss_cnx, test_accuracy_cnx = model_cnx.evaluate(test_ds_cnx, verbose=1)
y_pred_cnx = []
y_true_cnx = []

for images, labels in test_ds_cnx:
    preds = model_cnx.predict(images, verbose=0)
    y_pred_cnx.extend(np.argmax(preds, axis=1))
    y_true_cnx.extend(labels.numpy())

y_pred_cnx = np.array(y_pred_cnx)
y_true_cnx = np.array(y_true_cnx)

cm_cnx = confusion_matrix(y_true_cnx, y_pred_cnx)

plt.figure(figsize=(12, 10))
sns.heatmap(cm_cnx, annot=True, fmt='d', cmap='Greens', 
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Count'})
plt.title('ConvNeXtTiny - Confusion Matrix', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=12, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

comparison_data = {
    'Model': ['EfficientNetB0', 'ConvNeXtTiny'],
    'Parameters': [f"{model_eff.count_params():,}", f"{model_cnx.count_params():,}"],
    'Test Accuracy': [f"{test_accuracy_eff*100:.2f}%", f"{test_accuracy_cnx*100:.2f}%"],
    'Test Loss': [f"{test_loss_eff:.4f}", f"{test_loss_cnx:.4f}"]
}

print("\n{:<20} {:<20} {:<20} {:<20}".format('Model', 'Parameters', 'Test Accuracy', 'Test Loss'))
print("=" * 80)
for i in range(len(comparison_data['Model'])):
    print("{:<20} {:<20} {:<20} {:<20}".format(
        comparison_data['Model'][i],
        comparison_data['Parameters'][i],
        comparison_data['Test Accuracy'][i],
        comparison_data['Test Loss'][i]
    ))

# Side-by-side accuracy comparison
fig, ax = plt.subplots(figsize=(10, 6))
models = ['EfficientNetB0', 'ConvNeXtTiny']
accuracies = [test_accuracy_eff * 100, test_accuracy_cnx * 100]
colors = ['#2E86AB', '#06A77D']

bars = ax.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=2)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}%',
            ha='center', va='bottom', fontsize=14, fontweight='bold')

ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Model Comparison - Test Accuracy', fontsize=14, fontweight='bold')
ax.set_ylim([0, 100])
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

model_eff.save('efficientnetb0_skin_disease_final.keras')
model_eff.save('efficientnetb0_skin_disease_final.h5')
model_cnx.save('convnexttiny_skin_disease_final.keras')
model_cnx.save('convnexttiny_skin_disease_final.h5')

