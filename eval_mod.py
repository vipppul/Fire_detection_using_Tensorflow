import os
from pathlib import Path
from f1_callback import F1History

# 🔧 PATHS
BASE_DIR = Path(r"C:\Users\Asus\Desktop\Fire Detection\fire_dataset")  # raw dataset (fire/  no_fire/)
WORK_DIR = Path.cwd() / "fire_exp"   # where artefacts will be stored

# 🔧 DATA & TRAINING
IMG_SIZE     = (128, 128)
BATCH_SIZE   = 32
EPOCHS       = 18
K_FOLDS      = 5          # for cross‑validation
SEED         = 1337
DO_CROSS_VAL = True        # toggle CV; if False → simple 80/20 split
DO_FINAL_FIT = True        # train once on 90 % after CV and eval on test 10 %

# 🔧 MODEL CHOICE
USE_MOBILENET = True        # True → MobileNetV2 TL; False → custom small CNN
FREEZE_BASE    = True        # if using TL, freeze base during first phase

# 🔧 CALLBACKS
USE_EARLYSTOP  = True
PATIENCE_ES    = 5
USE_REDUCE_LR  = True
PATIENCE_RLROP = 3

# Ensure work dirs exist
WORK_DIR.mkdir(parents=True, exist_ok=True)
(WORK_DIR / "plots").mkdir(exist_ok=True)
(WORK_DIR / "models").mkdir(exist_ok=True)
(WORK_DIR / "logs").mkdir(exist_ok=True)

# ────────────────────────────────────────────────────────────────────────────────
# 1. IMPORTS & SEEDING
# ────────────────────────────────────────────────────────────────────────────────
import random
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_score, recall_score, f1_score, accuracy_score, roc_auc_score)
import matplotlib.pyplot as plt
import pandas as pd

# Reproducibility
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

# ────────────────────────────────────────────────────────────────────────────────
# 2. DATA LOADING UTILITIES
# ────────────────────────────────────────────────────────────────────────────────

def gather_filepaths_and_labels(base_dir: Path):
    """Return list[file_path], list[label_int]"""
    fire_dir = base_dir / "fire"
    nofire_dir = base_dir / "no_fire"
    filepaths = []
    labels    = []
    for fp in fire_dir.glob("**/*"):
        if fp.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            filepaths.append(str(fp))
            labels.append(1)
    for fp in nofire_dir.glob("**/*"):
        if fp.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            filepaths.append(str(fp))
            labels.append(0)
    return filepaths, labels

FILEPATHS, LABELS = gather_filepaths_and_labels(BASE_DIR)
print(f"Total images found → {len(FILEPATHS):,}")

# ────────────────────────────────────────────────────────────────────────────────
# 3. DATASET BUILDERS
# ────────────────────────────────────────────────────────────────────────────────

def preprocess_image(path: tf.Tensor, label: tf.Tensor):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0  # normalize 0‑1
    return image, label

# Data augmentation layer (only on training)
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.1),
])

def make_dataset(paths, labels, batch_size, training: bool):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, l: preprocess_image(p, l), num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# ────────────────────────────────────────────────────────────────────────────────
# 4. MODEL BUILDERS
# ────────────────────────────────────────────────────────────────────────────────

def build_custom_cnn(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model


def build_mobilenet_model(input_shape):
    base = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                             include_top=False,
                                             weights='imagenet')
    base.trainable = not FREEZE_BASE
    model = tf.keras.Sequential([
        base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model


def compile_model(model: tf.keras.Model):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    return model

# ────────────────────────────────────────────────────────────────────────────────
# 5. TRAINING UTILITIES
# ────────────────────────────────────────────────────────────────────────────────

def callbacks_for_fold(fold_idx, val_ds):
    cb = []

    # 1) F1 tracker FIRST  ✅
    cb.append(F1History(val_ds))

    # 2) checkpoint / CSV / others AFTER
    cb.append(tf.keras.callbacks.ModelCheckpoint(
        WORK_DIR / "models" / f"best_fold{fold_idx}.keras",   # switched to .keras too
        monitor='val_accuracy', save_best_only=True, verbose=0
    ))
    cb.append(tf.keras.callbacks.CSVLogger(
        WORK_DIR / "logs" / f"history_fold{fold_idx}.csv", append=False
    ))
    if USE_EARLYSTOP:
        cb.append(tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=PATIENCE_ES, restore_best_weights=True))
    if USE_REDUCE_LR:
        cb.append(tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.3, patience=PATIENCE_RLROP, verbose=1))

    return cb


# ────────────────────────────────────────────────────────────────────────────────
# 6. CROSS‑VALIDATION LOOP
# ────────────────────────────────────────────────────────────────────────────────

def run_cross_validation(paths, labels):
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    results = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(paths, labels), start=1):
        print(f"\n🟡 Fold {fold}/{K_FOLDS} — Train={len(train_idx)}, Val={len(val_idx)}")
        train_ds = make_dataset(np.array(paths)[train_idx], np.array(labels)[train_idx], BATCH_SIZE, training=True)
        val_ds   = make_dataset(np.array(paths)[val_idx],   np.array(labels)[val_idx],   BATCH_SIZE, training=False)

        # Build fresh model
        input_shape = IMG_SIZE + (3,)
        if USE_MOBILENET:
            model = build_mobilenet_model(input_shape)
        else:
            model = build_custom_cnn(input_shape)
        model = compile_model(model)

        h = model.fit(train_ds,
                      epochs=EPOCHS,
                      validation_data=val_ds,
                      callbacks=callbacks_for_fold(fold, val_ds),
                      verbose=2)

        # Evaluate best weights
        best_model = tf.keras.models.load_model(WORK_DIR / "models" / f"best_fold{fold}.keras")
        val_loss, val_acc, val_prec, val_rec, val_auc = best_model.evaluate(val_ds, verbose=0)
        val_pred_prob = best_model.predict(val_ds, verbose=0).ravel()
        val_pred      = (val_pred_prob > 0.5).astype(int)
        y_true        = np.concatenate([y.numpy() for _, y in val_ds])
        f1  = f1_score(y_true, val_pred)
        results.append(dict(fold=fold, acc=val_acc, precision=val_prec, recall=val_rec, f1=f1, auc=val_auc))

        # Plot training curves per fold
                # Plot training curves per fold
        plot_learning_curves(h.history, fold)

    # Final: plot best fold + CV mean±std
    plot_best_fold_and_cv_summary()
    return pd.DataFrame(results)


# ────────────────────────────────────────────────────────────────────────────────
# 7. PLOTTING UTILITIES
# ────────────────────────────────────────────────────────────────────────────────

def plot_learning_curves(hist, fold):
    plt.figure(figsize=(8,4))
    for metric in ['accuracy', 'loss', 'precision', 'recall']:
        plt.plot(hist[metric], label=f"train_{metric}")
        plt.plot(hist[f"val_{metric}"], label=f"val_{metric}")
        plt.title(f"Fold {fold} — {metric.capitalize()}")
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.legend()
        plt.tight_layout()
        plt.savefig(WORK_DIR / "plots" / f"fold{fold}_{metric}.png")
        plt.clf()
    plt.close('all')

def plot_best_fold_and_cv_summary():
    logs_dir = WORK_DIR / "logs"
    history_files = sorted(logs_dir.glob("history_fold*.csv"))
    histories = [pd.read_csv(f) for f in history_files]

    # Determine whether to use 'val_f1' or 'f1'
    col_f1 = 'val_f1' if 'val_f1' in histories[0].columns else 'f1'

    # Find best fold by highest final val_f1 (or f1)
    best_idx = max(range(len(histories)), key=lambda i: histories[i][col_f1].iloc[-1])
    best_df = histories[best_idx]

    # Plot best fold training vs validation metrics
    for base in ['accuracy', 'loss', 'precision', 'recall', 'f1']:
        train_col = base
        val_col = f"val_{base}"

        if val_col not in best_df.columns:
            continue  # Skip if no val_<metric> exists

        plt.figure()
        if train_col in best_df.columns:
            plt.plot(best_df[train_col], label='Train')
        plt.plot(best_df[val_col], label='Val')
        plt.title(f'Best Fold — {base.title()}')
        plt.xlabel('Epoch'); plt.ylabel(base.title())
        plt.legend(); plt.grid(True)
        plt.savefig(WORK_DIR / "plots" / f"best_fold_{base}.png")
        plt.close()

    # 📊 Aggregate mean±std plots for validation metrics
    for metric in ['val_accuracy', 'val_loss', 'val_precision', 'val_recall', 'val_f1']:
        if metric not in histories[0].columns:
            continue

        max_len = max(len(df[metric]) for df in histories)
        vals = []
        for df in histories:
            arr = df[metric].values
            padded = np.concatenate([arr, np.full(max_len - len(arr), np.nan)])
            vals.append(padded)
        vals = np.vstack(vals)

        mean = np.nanmean(vals, axis=0)
        std  = np.nanstd(vals, axis=0)

        plt.figure()
        plt.plot(mean, label=f'Mean {metric}')
        plt.fill_between(range(max_len), mean - std, mean + std, alpha=0.3, label='±1 std')
        plt.title(f'CV Mean±Std — {metric.replace("val_", "").title()}')
        plt.xlabel('Epoch'); plt.ylabel(metric.replace("val_", "").title())
        plt.legend(); plt.grid(True)
        plt.savefig(WORK_DIR / "plots" / f"cv_summary_{metric}.png")
        plt.close()


# ────────────────────────────────────────────────────────────────────────────────
# 8. FINAL MODEL ON 90 % (OPTIONAL)
# ────────────────────────────────────────────────────────────────────────────────

def train_full_and_test(train_paths, train_labels, test_paths, test_labels):
    train_ds = make_dataset(train_paths, train_labels, BATCH_SIZE, training=True)
    test_ds  = make_dataset(test_paths,  test_labels,  BATCH_SIZE, training=False)

    input_shape = IMG_SIZE + (3,)
    model = build_mobilenet_model(input_shape) if USE_MOBILENET else build_custom_cnn(input_shape)
    model = compile_model(model)

    val_ds_for_f1 = make_dataset(test_paths, test_labels, BATCH_SIZE, training=False)
    cb = callbacks_for_fold("final", val_ds_for_f1)

    history = model.fit(train_ds, epochs=EPOCHS, callbacks=cb, verbose=2)
    pd.DataFrame(history.history).to_csv(WORK_DIR / "logs" / "history_final.csv", index=False)

    for metric in ['accuracy', 'loss', 'precision', 'recall', 'val_f1']:
        plt.figure()
        plt.plot(history.history[metric], label='Train')
        plt.title(f'Final Model — {metric.title()}')
        plt.xlabel('Epoch'); plt.ylabel(metric.title()); plt.grid(True); plt.legend()
        plt.savefig(WORK_DIR / "plots" / f"final_{metric}.png")


    # Evaluate on held‑out test set
    test_loss, test_acc, test_prec, test_rec, test_auc = model.evaluate(test_ds, verbose=0)
    y_true = np.concatenate([y for _, y in test_ds], axis=0)
    y_prob = model.predict(test_ds, verbose=0).ravel()
    y_pred = (y_prob > 0.5).astype(int)
    test_f1 = f1_score(y_true, y_pred)

    # Print classification report
    print("\nTest‑set Classification Report:\n", classification_report(y_true, y_pred, target_names=["no_fire","fire"]))

    # Confusion matrix plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4,4))
    plt.imshow(cm, cmap='Blues')
    plt.title('Confusion Matrix — Test Set')
    plt.xticks([0,1], ['no_fire','fire'])
    plt.yticks([0,1], ['no_fire','fire'])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i,j], ha='center', va='center', color='white' if cm[i,j]>cm.max()/2 else 'black')
    plt.tight_layout()
    plt.savefig(WORK_DIR / "plots" / "confusion_matrix_test.png")
    plt.close()

    # Log overall metrics
    metrics_df = pd.DataFrame([dict(split="test", acc=test_acc, precision=test_prec, recall=test_rec, f1=test_f1, auc=test_auc)])
    metrics_df.to_csv(WORK_DIR / "logs" / "test_metrics.csv", index=False)
    model.save(WORK_DIR / "models" / "final_full_model.keras")
    print(f"Final model saved to {WORK_DIR/'models'/ 'final_full_model.keras'}")

# ────────────────────────────────────────────────────────────────────────────────
# 9. MAIN
# ────────────────────────────────────────────────────────────────────────────────

def main():
    # Split into temporary train_val_pool and test set
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        FILEPATHS, LABELS, test_size=0.10, stratify=LABELS, random_state=SEED)
    print(f"Pool for CV : {len(train_val_paths):,} images | Test set : {len(test_paths):,} images")

    # Run K‑fold CV
    if DO_CROSS_VAL:
        cv_df = run_cross_validation(train_val_paths, train_val_labels)
        print("\nFold metrics summary:\n", cv_df)
        cv_df.to_csv(WORK_DIR / "logs" / "cv_metrics.csv", index=False)
        print("\nAggregate (mean ± std):")
        for m in ['acc','precision','recall','f1','auc']:
            print(f"{m.upper():9}: {cv_df[m].mean():.4f} ± {cv_df[m].std():.4f}")

    # Train final model on 90 % and eval on 10 % test
    if DO_FINAL_FIT:
        train_full_and_test(np.array(train_val_paths), np.array(train_val_labels),
                            np.array(test_paths), np.array(test_labels))

    print(f"All artefacts are in {WORK_DIR.resolve()}")

# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
