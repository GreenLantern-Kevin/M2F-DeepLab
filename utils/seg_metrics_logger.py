"""Persist metrics and modality diagnostics after every epoch."""
import csv
from pathlib import Path
import numpy as np


class SegMetricsLogger:
    DIAGNOSTICS = ('loss_seg', 'loss_edge', 'loss_consist', 'weighted_seg',
                   'weighted_edge', 'weighted_consist', 'dem_present',
                   'route_real', 'route_fhm', 'route_zero', 'dem_dropped', 'samples', 'dem_coverage',
                   'grad_dem', 'grad_fhm', 'grad_mef', 'grad_concat', 'dem_dropout_active', 'alpha', 'beta', 'gradient_batches')

    def __init__(self, exp_dir, csv_name='metrics.csv'):
        self.exp_dir = Path(exp_dir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.exp_dir / csv_name
        self.rows = []

    def log_epoch(self, epoch, lr, train_loss, train_pacc, val_loss, val_score, diagnostics=None):
        row = dict(epoch=epoch, lr=lr, train_loss=train_loss, train_pacc=train_pacc, val_loss=val_loss)
        for short, key in [('pacc', 'Overall Acc'), ('macc', 'Mean Acc'), ('miou', 'Mean IoU'), ('fwiou', 'FreqW Acc')]:
            row['val_' + short] = float(val_score[key])
        for class_id, label in [(0, 'bg'), (1, 'ls')]:
            for short, key in [('iou', 'Class IoU'), ('prec', 'Class Precision'), ('rec', 'Class Recall'),
                               ('f1', 'Class F1'), ('mcc', 'Class MCC')]:
                row[f'val_{label}_{short}'] = float(val_score[key][class_id])
        row.update({key: (diagnostics or {}).get(key, '') for key in self.DIAGNOSTICS})
        row['val_dem_present'] = val_score.get('DEM Present', '')
        row['paired_samples'] = val_score.get('Paired Samples', '')
        for label, key in [('real', 'Paired Real'), ('missing', 'Paired Missing')]:
            score = val_score.get(key, {})
            row[f'paired_{label}_miou'] = score.get('Mean IoU', '')
            row[f'paired_{label}_ls_iou'] = score.get('Class IoU', {}).get(1, '')
        new_file = not self.csv_path.exists()
        with self.csv_path.open('a', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if new_file:
                writer.writeheader()
            writer.writerow(row)
        self.rows.append(row)
        cm = val_score.get('Confusion Matrix')
        if cm is not None:
            np.save(self.exp_dir / 'confusion_matrix_latest.npy', cm)
            self._plot_confusion(cm, epoch)

    def _plot_confusion(self, cm, epoch):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        den = cm.sum(axis=1, keepdims=True)
        normal = np.divide(cm, den, out=np.zeros_like(cm, dtype=float), where=den != 0)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(normal, cmap='Blues', vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{normal[i, j]:.2%}', ha='center', va='center',
                        color='white' if normal[i, j] > .5 else 'black')
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Background', 'Landslide'],
               yticklabels=['Background', 'Landslide'], xlabel='Predicted', ylabel='True', title=f'Epoch {epoch}')
        fig.tight_layout()
        fig.savefig(self.exp_dir / 'confusion_matrix_latest.png', dpi=150)
        plt.close(fig)

    def save_plots(self):
        if not self.rows:
            return
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        groups = {'loss_curve': ('train_loss', 'val_loss'), 'pacc_curve': ('train_pacc', 'val_pacc'),
                  'val_metrics_curve': ('val_macc', 'val_miou'),
                  'landslide_specific_metrics': ('val_ls_iou', 'val_ls_prec', 'val_ls_rec', 'val_ls_f1', 'val_ls_mcc'),
                  'branch_gradients': ('grad_dem', 'grad_fhm', 'grad_mef', 'grad_concat')}
        for name, keys in groups.items():
            fig, ax = plt.subplots()
            for key in keys:
                ax.plot([r['epoch'] for r in self.rows], [r[key] for r in self.rows], label=key)
            ax.set_xlabel('epoch')
            ax.legend()
            ax.grid(True)
            fig.tight_layout()
            fig.savefig(self.exp_dir / (name + '.png'), dpi=150)
            plt.close(fig)
