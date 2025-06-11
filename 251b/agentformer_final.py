# agentformer_scene_aware.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from transformers import get_cosine_schedule_with_warmup


# Load data
train_npz = np.load('./train.npz')
train_data = train_npz['data']
test_npz  = np.load('./test_input.npz')
test_data  = test_npz['data']

device = torch.device('cuda')

class TrajectoryDatasetTrain(torch.utils.data.Dataset):
    def __init__(self, data, scale=10.0, augment=True):
        self.data = data
        self.scale = scale
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        scene = self.data[idx]
        hist = scene[:, :50, :].copy()
        future = torch.tensor(scene[0, 50:, :2].copy(), dtype=torch.float32)

        if self.augment:
            if np.random.rand() < 0.5:
                theta = np.random.uniform(-np.pi, np.pi)
                R = np.array([[np.cos(theta), -np.sin(theta)],
                              [np.sin(theta),  np.cos(theta)]], dtype=np.float32)
                hist[..., :2] = hist[..., :2] @ R
                hist[..., 2:4] = hist[..., 2:4] @ R
                future = future @ R
            if np.random.rand() < 0.5:
                hist[..., 0] *= -1
                hist[..., 2] *= -1
                future[:, 0] *= -1

        origin = hist[0, 49, :2].copy().astype(np.float32)
        hist[..., :2] -= origin
        future -= origin

        hist[..., :4] /= self.scale
        future /= self.scale

        return Data(
            x=torch.tensor(hist, dtype=torch.float32),
            y=future,
            origin=torch.tensor(origin, dtype=torch.float32).unsqueeze(0),
            scale=torch.tensor(self.scale, dtype=torch.float32),
        )

class TrajectoryDatasetTest(torch.utils.data.Dataset):
    def __init__(self, data, scale=10.0):
        self.data = data
        self.scale = scale

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        scene = self.data[idx]
        hist = scene.copy()

        origin = hist[0, 49, :2].copy().astype(np.float32)
        hist[..., :2] -= origin
        hist[..., :4] /= self.scale

        return Data(
            x=torch.tensor(hist, dtype=torch.float32),
            origin=torch.tensor(origin, dtype=torch.float32).unsqueeze(0),
            scale=torch.tensor(self.scale, dtype=torch.float32),
        )
    
class AttentionPool(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, 1)

    def forward(self, x):
        weights = torch.softmax(self.query(x), dim=1)
        return torch.sum(weights * x, dim=1)
    
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=3)

    def forward(self, x):
        return self.transformer(x)


    
def trajectory_loss(pred, gt):
    pos_loss = nn.functional.mse_loss(pred, gt)
    vel_loss = nn.functional.mse_loss(pred[:, 1:] - pred[:, :-1], gt[:, 1:] - gt[:, :-1])
    final_loss = nn.functional.l1_loss(pred[:, -1], gt[:, -1])
    return pos_loss + 0.5 * vel_loss + 2.0 * final_loss


class CrossAttentionDecoder(nn.Module):
    def __init__(self, embed_dim, output_len):
        super().__init__()
        self.query_embed = nn.Parameter(torch.randn(1, output_len, embed_dim) * 0.01)
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=16, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=3)
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Linear(embed_dim, 2)

    def forward(self, memory, history):
        B = memory.size(0)
        query = self.query_embed.expand(B, -1, -1)
        out = self.decoder(query, history, memory_key_padding_mask=None)
        out = self.linear(out)
        out = self.relu(out)
        out = self.norm(out)
        return self.out_proj(out)


class AgentFormerSceneAware(nn.Module):
    def __init__(self, input_dim=6, embed_dim=128, obj_type_embed_dim=8):
        super().__init__()
        self.obj_type_embed = nn.Embedding(10, obj_type_embed_dim)
        self.input_proj = nn.Linear(input_dim - 1 + obj_type_embed_dim, embed_dim)
        self.encoder = TransformerEncoderBlock(embed_dim)
        self.attn_pool = AttentionPool(embed_dim)

        # bottleneck aggregator
        self.aggregator = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        self.goal_proj = nn.Linear(embed_dim, embed_dim)
        self.decoder = CrossAttentionDecoder(embed_dim, output_len=60)
        self.pos_embed = nn.Parameter(torch.randn(1, 50, embed_dim) * 0.01)

    def forward(self, x):
        B = x.shape[0]
        object_type = x[..., 5].long()
        obj_embed = self.obj_type_embed(object_type)
        x_features = x[..., :5]
        x = torch.cat([x_features, obj_embed], dim=-1)
        x = self.input_proj(x)
        x = x + self.pos_embed

        x = x.view(B * 50, 50, -1)
        enc = self.encoder(x)
        enc = enc.mean(dim=1)
        enc = enc.view(B, 50, -1)


        goal_hint = enc[:, -1, :]
        goal_token = self.goal_proj(goal_hint).unsqueeze(1)
        history_plus_goal = torch.cat([enc, goal_token], dim=1)

        scene_encoding = self.attn_pool(enc)
        scene_feat = self.aggregator(scene_encoding).unsqueeze(1)
        out = self.decoder(scene_feat, history_plus_goal)
        return out



def train_model(model, num_epochs=100):
    scale = 7.0
    N = len(train_data)
    val_size = int(0.1 * N)
    train_size = N - val_size

    train_dataset = TrajectoryDatasetTrain(train_data[:train_size], scale=scale, augment=True)
    val_dataset = TrajectoryDatasetTrain(train_data[train_size:], scale=scale, augment=False)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=lambda x: Batch.from_data_list(x))
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=lambda x: Batch.from_data_list(x))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    total_steps = len(train_dataloader) * num_epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    train_losses, val_losses = [], []
    patience = 10
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    loss_fn = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1} [Train]"):
            x = batch.x.view(batch.num_graphs, 50, 50, 6).to(device)
            y = batch.y.view(batch.num_graphs, 60, 2).to(device)

            pred = model(x)
            loss = trajectory_loss(pred, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_total_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1} [Val]"):
                x = batch.x.view(batch.num_graphs, 50, 50, 6).to(device)
                y = batch.y.view(batch.num_graphs, 60, 2).to(device)

                pred = model(x)
                loss = trajectory_loss(pred, y)
                val_total_loss += loss.item()

        avg_val_loss = val_total_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)
        #torch.save(model.state_dict(), f'checkpoints2/agentformer2_epoch_{epoch}.pt')

        #if avg_val_loss - avg_train_loss < 0.03:  #
        if val_total_loss < best_val_loss:
            print(f"update best val loss: {val_total_loss:.4f} (previous: {best_val_loss:.4f})")
            best_val_loss = val_total_loss
            epochs_without_improvement = 0
            os.makedirs("checkpoints2", exist_ok=True)
            torch.save(model.state_dict(), 'checkpoints2/agent_former_best__.pt')
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1} with best validation loss: {best_val_loss:.4f}")
                break

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Score:{val_total_loss:.4f}")

    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_curve.png")
    plt.close()

    state_dict = torch.load('checkpoints2/agent_former_best.pt', map_location=device)
    model.load_state_dict(state_dict)
    return model

def test_model(model, test_data, scale=7.0):
    test_dataset = TrajectoryDatasetTest(test_data, scale=scale)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=lambda x: Batch.from_data_list(x))

    model.eval()
    pred_list = []
    with torch.no_grad():
        for batch in test_dataloader:
            batch = batch.to(device)
            #print("Batch shape before model:", batch.x.shape)
            B = batch.num_graphs
            try:
                x = batch.x.view(B, 50, 50, 6).to(device)
            except RuntimeError:
                print(f"Failed to reshape batch.x of shape {batch.x.shape} into ({B}, 50, 50, 6)")
                raise

            pred_norm = model(x)

            #print(pred_norm)
            pred = pred_norm * batch.scale.view(-1, 1, 1) + batch.origin.unsqueeze(1)
            pred_list.append(pred.cpu().numpy())

        pred_list = np.concatenate(pred_list, axis=0)  # (N, 60, 2)
        pred_output = pred_list.reshape(-1, 2)  # (N * 60, 2)

    output_df = pd.DataFrame(pred_output, columns=['x', 'y'])
    output_df.index.name = 'index'
    output_df.to_csv('submission.csv', index=True)

    print("Submission saved to 'submission.csv'")


if __name__ == "__main__":
    model = AgentFormerSceneAware().to(device)
    #model_path = 'checkpoints2/agent_former_best.pt'
    model_path = 'checkpoints2/agent_.pt'
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        try:
            model.load_state_dict(state_dict)
            print(f"Model loaded from {model_path}")
        except Exception:
            print("Failed to load model state_dict. Initializing a new model.")
            model = AgentFormerSceneAware().to(device)
    model = train_model(model=model, num_epochs=10)
    #test_model(model, test_data)