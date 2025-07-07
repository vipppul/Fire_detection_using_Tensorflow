# f1_callback.py
import tensorflow as tf
import numpy as np

class F1History(tf.keras.callbacks.Callback):
    def __init__(self, val_data):
        super().__init__()
        self.val_data = val_data
        self.f1s = []

    def on_epoch_end(self, epoch, logs=None):
        y_true, y_pred = [], []
        for batch_x, batch_y in self.val_data:
            preds = self.model(batch_x, training=False).numpy().ravel()
            y_pred.extend(preds)
            y_true.extend(batch_y)
        y_pred_bin = (np.array(y_pred) > 0.5).astype(int)
        y_true = np.array(y_true).astype(int)
        f1 = 2 * ( (y_pred_bin & y_true).sum() ) / ( (y_pred_bin.sum() + y_true.sum()) or 1 )
        self.f1s.append(f1)
        logs = logs or {}
        # logs["f1"]= f1
        logs["val_f1"] = f1   # ← add the “val_” prefix
        self.f1s.append(f1)
