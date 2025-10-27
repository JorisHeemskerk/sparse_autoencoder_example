import csv
import os
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm


def train_sparse_autoencoder(
    model: torch.nn.Module, 
    optimizer: torch.optim, 
    train_dataloader: torch.utils.data.DataLoader, 
    val_dataloader: torch.utils.data.DataLoader,
    num_epochs: int, 
    device: torch._prims_common.DeviceLikeType,
    l1_coeff: float,
    replacement: bool=False,
    save_model: bool=True,
    output_folder_name: str=None,
    output_file_name: str=None,
    save_loss: bool=True,
)-> None:
    """ Sparse autoencoder training loop

        @param model: SparseAutoencoder model (see sparse_autoencoder.py).
        @type  model: torch.nn.Module
        @param optimizer: Optimizer used to train.
        @type  optimizer: torch.optim
        @param train_dataloader: Data container for training data.
        @type  train_dataloader: torch.utils.data.DataLoader
        @param val_dataloader: Data container for validation data.
        @type  val_dataloader: torch.utils.data.DataLoader
        @param num_epochs: Number of iterations.
        @type  num_epochs: int
        @param device: Either 'cuda' or 'cpu'. Add :i with i being index of device in case of multiple options.
        @type  device: torch._prims_common.DeviceLikeType
        @param l1_coeff: L1 learning coefficient.
        @type  l1_coeff: float
        @param replacement: Whether or not to repeat datapoints during training. (DEFAULT=False)
        @type  replacement: bool
        @param save_model: True if model should be saved to file (must provide next 2 arugments too). (DEFAULT=True)
        @type  save_model: bool
        @param output_folder_name: Output folder. Create this folder if it does not exist yet. (DEFAULT=None)
        @type  output_folder_name: str
        @param output_file_name: filename of output file. (DEFAULT=None)
        @type  output_file_name: str
        @param save_loss: True if loss plot should be saved.
        @type  save_loss: bool
    """

    # enforce that output_folder_name and output_file_name are present if save is True
    assert (save_model and output_folder_name is not None and  output_file_name is not None) or not save_model, \
        f"Save is {save_model}, and either output_folder_name ({output_folder_name}) or " \
        f"output_file_name ({output_file_name}) is None."

    train_loss_history = []
    val_loss_history = []
    criterion = torch.nn.MSELoss(reduction='mean')  # MSE averaged per element

    # Parameters for dead neuron resampling.
    resample_steps = {25000, 50000, 75000, 100000, 125000, 150000}
    window_steps = 12500  # number of mini-batches to consider for dead neuron activity
    # Step counter
    global_step = 0
    # Initialize a flag vector for the hidden neurons.
    # Size is hidden_dim, taken from the encoder's output dimension.
    hidden_dim = model.encoder.out_features
    fired_flag = torch.zeros(hidden_dim, dtype=torch.bool, device=device)

    # Helper: Reset Adam state for a given parameter tensor at specified indices.
    def reset_adam_state(param: torch.nn.Parameter, indices: torch.Tensor, dim: int):
        # For Adam, the state for param includes buffers 'exp_avg' and 'exp_avg_sq'.
        state = optimizer.state.get(param, None)
        if state is None:
            return
        for key in ['exp_avg', 'exp_avg_sq']:
            buf = state.get(key, None)
            if buf is not None:
                # If the state tensor shape aligns with the given dimension, zero-out the affected indices.
                # (For encoder.weight, dim=0; for decoder.weight, dim=1; for encoder.bias, dim=0)
                if buf.dim() > dim:
                    buf.index_fill_(dim, indices, 0.0)
                else:
                    buf[indices] = 0.0

    # Resampling procedure.
    def resample_dead_neurons():
        nonlocal fired_flag
        dead_neuron_idx = (fired_flag == False).nonzero(as_tuple=False).squeeze()  # indices of neurons that never fired
        if dead_neuron_idx.numel() == 0:
            fired_flag.zero_()
            return

        # Step 1: Draw a random subset of inputs (size 21062 (~10% of data) or the maximum available)
        dataset = train_dataloader.dataset
        total_inputs = len(dataset)
        subset_size = min(21062, total_inputs)
        # Randomly sample indices from the dataset.
        subset_indices = torch.randperm(total_inputs)[:subset_size].tolist()
        # Collect the subset inputs into a tensor. Assumes dataset supports indexing and returns a tensor.
        inputs_subset = torch.stack([dataset[i][0].clone().detach().to(device) for i in subset_indices])
        
        # Step 2: Compute the per-input reconstruction loss (without reduction) for the subset.
        model.eval()
        with torch.no_grad():
            outputs, activations = model(inputs_subset)
            # Compute squared error per sample (sum over features)
            # losses = criterion(outputs, inputs_subset) + l1_coeff * torch.sum(torch.abs(activations))
            rec_loss = torch.sum((outputs - inputs_subset) ** 2, dim=1)
            # per-sample L1 sparsity penalty
            l1_loss = l1_coeff * torch.sum(torch.abs(activations), dim=1)
            # combine
            losses = rec_loss + l1_loss
            # Compute probability weights proportional to square of the loss.
            probs = losses ** 2
            probs /= torch.sum(probs)
        
        # Step 3: For each dead neuron, sample an input and reinitialize.
        # Compute average encoder weight norm over alive neurons.
        alive_idx = (fired_flag == True).nonzero(as_tuple=False).squeeze()
        if alive_idx.numel() > 0:
            alive_weights = model.encoder.weight[alive_idx, :]  # shape: (num_alive, input_dim)
            avg_alive_norm = torch.mean(alive_weights.norm(p=2, dim=1))
        else:
            avg_alive_norm = 1.0  # default if none are alive

        # If only 1 neuron, it needs to be put in the shape of a list.
        if dead_neuron_idx.dim() == 0:
            dead_neuron_idx = [dead_neuron_idx]
        else:
            dead_neuron_idx = dead_neuron_idx.tolist()
        for neuron in dead_neuron_idx:
            # Sample one input index according to probabilities.
            sample_idx = torch.multinomial(probs, 1).item()
            input_vec = inputs_subset[sample_idx]  # shape: (input_dim,)
            # Normalize input vector to unit L2 norm.
            new_vector = input_vec / (input_vec.norm(p=2) + 1e-8)
            # Update the decoder weight column corresponding to this neuron.
            # decoder.weight shape: (input_dim, hidden_dim)
            with torch.no_grad():
                model.decoder.weight[:, neuron].copy_(new_vector)
            # For the encoder weight: renormalize input vector so that its norm equals (avg_alive_norm * 0.2).
            new_enc_vector = new_vector * (avg_alive_norm * 0.2)
            with torch.no_grad():
                model.encoder.weight[neuron, :].copy_(new_enc_vector)
                model.encoder.bias[neuron] = 0.0

            # Reset Adam state for the modified parameters.
            # For encoder.weight (reset row 'neuron', dim=0), encoder.bias (index 'neuron'),
            # and for decoder.weight (reset column 'neuron', dim=1).
            neuron_idx_tensor = torch.tensor([neuron], device=device)
            reset_adam_state(model.encoder.weight, neuron_idx_tensor, dim=0)
            reset_adam_state(model.encoder.bias, neuron_idx_tensor, dim=0)
            reset_adam_state(model.decoder.weight, neuron_idx_tensor, dim=1)

        # After resampling, reset the fired flag.
        fired_flag.zero_()

    pbar = tqdm(range(1, num_epochs + 1), desc="training the model")
    for epoch in pbar: # epoch loop
        model.train()
        epoch_loss = 0.0
        for inputs in train_dataloader:
            global_step += 1

            inputs = inputs[0].to(device)  # Extract tensor from tuple
            optimizer.zero_grad()
            
            # Forward pass: get reconstruction and hidden activations
            outputs, activations = model(inputs)
            
            # Update fired flag: mark neurons that fired (activation > 0) in this mini-batch.
            batch_fired = (activations > 0).any(dim=0)
            fired_flag |= batch_fired

            # Reconstruction loss: mean squared error between inputs and outputs.
            mse_loss = criterion(outputs, inputs)
            
            # Sparsity loss: L1 penalty on the hidden activations.
            l1_loss = l1_coeff * torch.sum(torch.abs(activations))
            
            # Total loss is the sum of reconstruction loss and sparsity loss.
            loss = mse_loss + l1_loss
            
            loss.backward()
            optimizer.step()
            
            # Normalize the decoder's weight columns after each update.
            model.normalize_decoder_weights()
            
            epoch_loss += loss.item()

            # Check if it's time to perform dead neuron resampling.
            if global_step in resample_steps:
                resample_dead_neurons()
            # Optionally, if you want to reset the fired flag every window_steps (even if not resampling),
            # uncomment the following:
            if global_step % window_steps == 0:
                fired_flag.zero_()
            # Only train 1 batch if replacement is False.
            if not replacement:
                break

        avg_train_loss = epoch_loss / (len(train_dataloader) if replacement else 1)
        train_loss_history.append(avg_train_loss)

        # Evaluate on the validation set.
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_inputs in val_dataloader:
                val_inputs = val_inputs[0].to(device)
                outputs, activations = model(val_inputs)
                mse_loss = criterion(outputs, val_inputs)
                l1_loss = l1_coeff * torch.sum(torch.abs(activations))
                loss = mse_loss + l1_loss
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_dataloader)
        val_loss_history.append(avg_val_loss)

        g = model.parameters()
        pbar.set_postfix({
            "\x1b[33;20mAvg train loss" : f"{avg_train_loss:.4f}", 
            "Avg val loss" : f"{avg_val_loss:.4f}",
            "Learning rate" : f"{optimizer.param_groups[0]['lr'] / (next(g).grad.std().item() + 1e-8):.10f}\x1b[0m"
        })
            
    if save_model:
        if not os.path.exists(output_folder_name):
            os.makedirs(output_folder_name)
        else:
            print(
                f"WARNING: output folder {output_folder_name} already exists. Any duplicate files will be overwritten."
            )
        torch.save(model, f"{output_folder_name}{output_file_name}")

    if save_loss:
        with open(f"{output_folder_name}loss_history.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss"])
            for epoch_idx, (t_loss, v_loss) in enumerate(zip(train_loss_history, val_loss_history), start=1):
                writer.writerow([epoch_idx, t_loss, v_loss])

        plt.plot(train_loss_history, label="Train")
        plt.plot(val_loss_history, label="Val")
        plt.xlabel("Epoch")
        plt.ylabel("Average loss")
        plt.title("Training and Validation Loss per Epoch")
        plt.legend()
        plt.savefig(output_folder_name + f"Loss_graph.png")
        plt.close()
