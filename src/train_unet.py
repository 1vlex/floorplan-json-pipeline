# src/train_unet_smp.py
import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import segmentation_models_pytorch as smp
from tqdm.auto import tqdm


class FloorplanDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str) -> None:
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)

        exts = {".jpg", ".jpeg", ".png"}
        self.samples: List[Tuple[Path, Path]] = []

        for img_path in self.images_dir.iterdir():
            if img_path.suffix.lower() not in exts:
                continue
            mask_name = img_path.stem + "_mask.png"
            mask_path = self.masks_dir / mask_name
            if mask_path.exists():
                self.samples.append((img_path, mask_path))

        if not self.samples:
            raise RuntimeError(
                f"No image/mask pairs found in {images_dir} / {masks_dir}"
            )

        # ImageNet-нормализация для энкодера ResNet
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.samples[idx]

        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"Cannot read image: {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = img_rgb.astype(np.float32) / 255.0  # HWC

        img = (img - self.mean) / self.std

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Cannot read mask: {mask_path}")
        mask = mask.astype(np.int64)  # 0..3

        img = np.transpose(img, (2, 0, 1))  # CHW

        img_t = torch.from_numpy(img)
        mask_t = torch.from_numpy(mask)

        return img_t, mask_t


def train_one_epoch(model, loader, optimizer, device, epoch: int):
    model.train()
    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0

    pbar = tqdm(loader, desc=f"Train {epoch}", leave=False)
    for imgs, masks in pbar:
        imgs = imgs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = ce(logits, masks)
        loss.backward()
        optimizer.step()

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

        pbar.set_postfix(loss=loss.item())

    return total_loss / max(1, total_samples)


@torch.no_grad()
def eval_one_epoch(model, loader, device, epoch: int, num_classes: int = 4):
    """
    Считаем:
      - средний CE loss,
      - mIoU по классам 1..(num_classes-1) (фон=0 игнорируем).
    """
    model.eval()
    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0

    inter = np.zeros(num_classes, dtype=np.float64)
    union = np.zeros(num_classes, dtype=np.float64)

    pbar = tqdm(loader, desc=f"Val {epoch}", leave=False)
    for imgs, masks in pbar:
        imgs = imgs.to(device)
        masks = masks.to(device)

        logits = model(imgs)
        loss = ce(logits, masks)

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

        pbar.set_postfix(loss=loss.item())

        preds = torch.argmax(logits, dim=1)  # B, H, W

        for cls in range(1, num_classes):  # 1..3, фон (0) не считаем
            pred_c = preds == cls
            mask_c = masks == cls

            if not mask_c.any() and not pred_c.any():
                continue

            inter_cls = torch.logical_and(pred_c, mask_c).sum().item()
            union_cls = torch.logical_or(pred_c, mask_c).sum().item()

            inter[cls] += inter_cls
            union[cls] += union_cls

    mean_loss = total_loss / max(1, total_samples)

    ious = []
    for cls in range(1, num_classes):
        if union[cls] > 0:
            ious.append(inter[cls] / union[cls])
    miou = float(np.mean(ious)) if ious else 0.0

    return mean_loss, miou


def main():
    parser = argparse.ArgumentParser(
        description="Train pretrained Unet (ResNet34 encoder) on cubicasa5k masks"
    )
    parser.add_argument("--train_images_dir", type=str, required=True)
    parser.add_argument("--train_masks_dir", type=str, required=True)
    parser.add_argument("--val_images_dir", type=str, required=True)
    parser.add_argument("--val_masks_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Кол-во эпох без улучшения val mIoU до early stopping",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # лог в файл
    log_path = out_dir / "train_log.txt"
    log_f = log_path.open("w", encoding="utf-8")
    log_f.write("epoch,train_loss,val_loss,val_mIoU,best_val_mIoU\n")

    # датасеты: train / valid из разных папок
    train_ds = FloorplanDataset(args.train_images_dir, args.train_masks_dir)
    val_ds = FloorplanDataset(args.val_images_dir, args.val_masks_dir)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    log_f.write(f"# Using device: {device}\n")

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=4,  # 0 bg, 1 wall, 2 door, 3 window
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_miou = -1.0
    best_epoch = -1
    epochs_no_improve = 0

    best_path = out_dir / "unet_resnet34_best.pth"
    last_path = out_dir / "unet_resnet34_last.pth"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss, val_miou = eval_one_epoch(
            model, val_loader, device, epoch, num_classes=4
        )

        line = (
            f"Epoch {epoch:03d}: "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_mIoU={val_miou:.4f}"
        )
        print(line)

        log_f.write(
            f"{epoch},{train_loss:.6f},{val_loss:.6f},{val_miou:.6f},{best_val_miou:.6f}\n"
        )
        log_f.flush()

        # сохраняем "последнюю" модель на всякий случай
        torch.save(
            {
                "model_state": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "val_mIoU": val_miou,
            },
            last_path,
        )

        # критерий лучшей модели: максимальный mIoU
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            best_epoch = epoch
            epochs_no_improve = 0

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_mIoU": val_miou,
                },
                best_path,
            )
            print(f"  -> new best mIoU={val_miou:.4f}, saved to {best_path}")
        else:
            epochs_no_improve += 1
            print(
                f"  -> no improvement for {epochs_no_improve} epoch(s) "
                f"(best mIoU={best_val_miou:.4f} at epoch {best_epoch})"
            )
            if epochs_no_improve >= args.patience:
                print(
                    f"Early stopping: no mIoU improvement for {args.patience} epochs."
                )
                break

    log_f.write(
        f"# Training done. Best mIoU={best_val_miou:.4f} at epoch {best_epoch}\n"
    )
    log_f.close()

    print(
        f"Training done. Best val_mIoU={best_val_miou:.4f}, "
        f"best epoch={best_epoch}, model={best_path}"
    )


if __name__ == "__main__":
    main()
