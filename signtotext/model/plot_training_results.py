import json
import matplotlib.pyplot as plt
import numpy as np

# Path to results
results_path = 'training_results.json'

with open(results_path, 'r') as f:
    results = json.load(f)

history = results['training_history']
metrics = results['performance_metrics']

epochs = history['epochs']
best_epoch = metrics['best_epoch']

# Helper for best epoch marker
def mark_best(ax, x, y, label):
    ax.axvline(x=best_epoch, color='gray', linestyle='--', alpha=0.5)
    ax.scatter([best_epoch], [y[best_epoch-1]], color='red', zorder=5, label=label)

# 1. Accuracy plot
plt.figure(figsize=(8, 5))
plt.plot(epochs, history['training_accuracy'], label='Train Accuracy', color='#1976d2', linestyle='-', marker='o')
plt.plot(epochs, history['validation_accuracy'], label='Val Accuracy', color='#ff9800', linestyle='--', marker='s')
plt.title('Model Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(0, 1.05)
plt.grid(True, linestyle=':')
mark_best(plt.gca(), epochs, history['validation_accuracy'], 'Best Val Acc')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('accuracy_curve.png')
plt.show()
plt.close()

# 2. Precision plot
plt.figure(figsize=(8, 5))
plt.plot(epochs, history['training_precision'], label='Train Precision', color='#388e3c', linestyle='-', marker='o')
plt.plot(epochs, history['validation_precision'], label='Val Precision', color='#f44336', linestyle='--', marker='s')
plt.title('Model Precision over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.ylim(0, 1.05)
plt.grid(True, linestyle=':')
mark_best(plt.gca(), epochs, history['validation_precision'], 'Best Val Prec')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('precision_curve.png')
plt.show()
plt.close()

# 3. Recall plot
plt.figure(figsize=(8, 5))
plt.plot(epochs, history['training_recall'], label='Train Recall', color='#7b1fa2', linestyle='-', marker='o')
plt.plot(epochs, history['validation_recall'], label='Val Recall', color='#0097a7', linestyle='--', marker='s')
plt.title('Model Recall over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.ylim(0, 1.05)
plt.grid(True, linestyle=':')
mark_best(plt.gca(), epochs, history['validation_recall'], 'Best Val Recall')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('recall_curve.png')
plt.show()
plt.close()

# 4. F1-score plot
plt.figure(figsize=(8, 5))
plt.plot(epochs, history['training_f1'], label='Train F1', color='#e65100', linestyle='-', marker='o')
plt.plot(epochs, history['validation_f1'], label='Val F1', color='#43a047', linestyle='--', marker='s')
plt.title('Model F1-score over Epochs')
plt.xlabel('Epoch')
plt.ylabel('F1-score')
plt.ylim(0, 1.05)
plt.grid(True, linestyle=':')
mark_best(plt.gca(), epochs, history['validation_f1'], 'Best Val F1')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('f1_curve.png')
plt.show()
plt.close()

# 5. Loss plot (improved style)
plt.figure(figsize=(8, 5))
plt.plot(epochs, history['training_loss'], label='Training Loss', color='#607d8b', linestyle='-', marker='o')
plt.plot(epochs, history['validation_loss'], label='Validation Loss', color='#d32f2f', linestyle='--', marker='s')
plt.title('Model Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, linestyle=':')
plt.legend()
plt.tight_layout()
plt.savefig('loss_curve.png')
plt.show()
plt.close()

# 6. Summary figure (all metrics as subplots)
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.flatten()
axs[0].plot(epochs, history['training_accuracy'], label='Train', color='#1976d2')
axs[0].plot(epochs, history['validation_accuracy'], label='Val', color='#ff9800')
axs[0].set_title('Accuracy')
axs[0].set_ylim(0, 1.05)
axs[0].legend()
axs[0].grid(True, linestyle=':')

axs[1].plot(epochs, history['training_precision'], label='Train', color='#388e3c')
axs[1].plot(epochs, history['validation_precision'], label='Val', color='#f44336')
axs[1].set_title('Precision')
axs[1].set_ylim(0, 1.05)
axs[1].legend()
axs[1].grid(True, linestyle=':')

axs[2].plot(epochs, history['training_recall'], label='Train', color='#7b1fa2')
axs[2].plot(epochs, history['validation_recall'], label='Val', color='#0097a7')
axs[2].set_title('Recall')
axs[2].set_ylim(0, 1.05)
axs[2].legend()
axs[2].grid(True, linestyle=':')

axs[3].plot(epochs, history['training_f1'], label='Train', color='#e65100')
axs[3].plot(epochs, history['validation_f1'], label='Val', color='#43a047')
axs[3].set_title('F1-score')
axs[3].set_ylim(0, 1.05)
axs[3].legend()
axs[3].grid(True, linestyle=':')

for ax in axs:
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
fig.suptitle('Model Metrics Overview', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig('metrics_overview.png')
plt.show()
plt.close()

# 7. Bar chart of final metrics (improved)
final_metrics = {
    'Train Accuracy': metrics['final_training_accuracy'],
    'Val Accuracy': metrics['final_validation_accuracy'],
    'Train Precision': metrics['final_training_precision'],
    'Val Precision': metrics['final_validation_precision'],
    'Train Recall': metrics['final_training_recall'],
    'Val Recall': metrics['final_validation_recall'],
    'Train F1': metrics['final_training_f1'],
    'Val F1': metrics['final_validation_f1'],
    'Train Loss': metrics['final_training_loss'],
    'Val Loss': metrics['final_validation_loss']
}

plt.figure(figsize=(12, 6))
colors = ['#1976d2', '#ff9800', '#388e3c', '#f44336', '#7b1fa2', '#0097a7', '#e65100', '#43a047', '#607d8b', '#d32f2f']
plt.bar(final_metrics.keys(), final_metrics.values(), color=colors)
plt.title('Final Model Metrics')
plt.ylabel('Value')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('final_metrics.png')
plt.show()
plt.close()

print('Best graphs saved: accuracy_curve.png, precision_curve.png, recall_curve.png, f1_curve.png, loss_curve.png, metrics_overview.png, final_metrics.png in the model folder.')
