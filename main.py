
import torch

from torch.utils.data import TensorDataset, DataLoader, random_split

from sparse_autoencoder import SparseAutoencoder
from training import train_sparse_autoencoder

DEVICE = "cpu"


if __name__ == "__main__":
    model = SparseAutoencoder(
            input_dim=128, 
            hidden_dim=512,
        ).to(DEVICE)
    
    X = torch.rand(1000, 128)
    dataset = TensorDataset(X)
    train_dataset, test_dataset = random_split(dataset, [800, 200])
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    
    train_sparse_autoencoder(
        model=model, 
        optimizer=torch.optim.Adam(model.parameters(), lr=0.00001), 
        train_dataloader=train_loader, 
        val_dataloader=val_loader, 
        num_epochs=100, 
        device=DEVICE,
        l1_coeff=0.000005,
        replacement=True,
        save_model=True,
        output_folder_name="output/",
        output_file_name= f"model.pt",
        save_loss=True,
    )
