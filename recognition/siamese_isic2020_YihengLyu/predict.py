import argparse, os
import torch
from torch.utils.data import DataLoader
from dataset import ISIC2020Singles, eval_transforms
from modules import SiameseNet

@torch.no_grad()
def evaluate_linear_probe(ckpt_path, csv, images, img_size=224, batch_size=64):
    """
    简易“线性探针”评估：冻结 backbone，基于验证集 embeddings 训练一个 Logistic 回归器，
    以获得单图二分类准确率（不是 pair-acc）。为了初版简单，这里用 torch 实现一个一层线性头。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(ckpt_path, map_location=device)
    args = ckpt["args"]

    model = SiameseNet(embedding_dim=args["embed_dim"], pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    ds = ISIC2020Singles(csv, images, transform=eval_transforms(img_size))
    n = len(ds)
    # 切 80/20 做个快速评估
    n_train = int(0.8 * n)
    idx_train = list(range(0, n_train))
    idx_test = list(range(n_train, n))
    ds_train = ISIC2020Singles(csv, images, transform=eval_transforms(img_size), ids_subset=[ds.df.iloc[i]['image_name'] for i in idx_train])
    ds_test  = ISIC2020Singles(csv, images, transform=eval_transforms(img_size), ids_subset=[ds.df.iloc[i]['image_name'] for i in idx_test])

    def embed_dataset(ds_local):
        loader = DataLoader(ds_local, batch_size=batch_size, shuffle=False, num_workers=2)
        feats, labels = [], []
        for x, y, _ in loader:
            x = x.to(device)
            z = model.backbone(x)  # (B, D)
            feats.append(z.cpu())
            labels.extend(y.numpy().tolist())
        return torch.cat(feats), torch.tensor(labels, dtype=torch.float32)

    Xtr, ytr = embed_dataset(ds_train)
    Xte, yte = embed_dataset(ds_test)

    # 训练一个 logistic 回归（sigmoid）
    W = torch.zeros(Xtr.shape[1], 1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=1e-2)
    for _ in range(300):
        logits = Xtr @ W + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.squeeze(1), ytr)
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        logits = Xte @ W + b
        pred = (torch.sigmoid(logits.squeeze(1)) > 0.5).float()
        acc = (pred == yte).float().mean().item()

    print(f"[Linear-probe Accuracy] {acc:.4f} on hold-out split using embeddings from {os.path.basename(ckpt_path)}")
    return acc

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--csv", type=str, default="ISIC_2020_Training_GroundTruth.csv")
    ap.add_argument("--images", type=str, default="train")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()
    evaluate_linear_probe(args.ckpt, args.csv, args.images, args.img_size, args.batch_size)
