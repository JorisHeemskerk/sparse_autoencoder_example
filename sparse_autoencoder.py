import torch
import torch.nn as nn


class SparseAutoencoder(nn.Module):
    """ Sparse autoencoder from Bricken, et al. """

    def __init__(self, input_dim: int, hidden_dim: int)-> None:
        """ One-hidden-layer sparse autoencoder

            The encoder uses weights W_e (e=encoder) and bias b_e,
            and the decoder uses weights W_d (d=decoder) (with unit-norm columns) and bias b_d.
            The autoencoder subtracts the decoder bias (b_d) from the inputs (pre-encoder bias).

            @param input_dim: Input dimensions (should be size of embeddings).
            @type  input_dim: int
            @param hidden_dim: Hidden dimensions (should be same number as desired amount of features).
            @type  hidden_dim: int
        """
        super(SparseAutoencoder, self).__init__()

        # Encoder parameters: W_e (size: hidden_dim x input_dim) and b_e (size: hidden_dim)
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)

        # Decoder parameters: W_d (size: input_dim x hidden_dim) and b_d (size: input_dim)
        # Note: We are not tying the weights between encoder and decoder.
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=True)

        # Initialize decoder weights with Kaiming Uniform, and ensure columns are unit norm.
        nn.init.kaiming_uniform_(self.encoder.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.decoder.weight, nonlinearity='linear')

        # Optionally, you can initialize bias to a fixed value (e.g., geometric median of the dataset).
        # For this demo, we initialize it to zeros.
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)

        # Put decoder data on unit norm
        self.normalize_decoder_weights()
        
    def forward(self, x: torch.Tensor)-> torch.Tensor:
        """ Forward pass

            @param x: input embedding batch
            @type  x: torch.Tensor

            @rtype: torch.Tensor
            @return: Encoded -> decoded data.
        """
        # Subtract decoder bias from the input (pre-encoder bias)
        # This ensures that the pre-encoder bias is the negative of the post-decoder bias.
        x_bar = x - self.decoder.bias
        
        # Encoder: f = ReLU(W_e * x_bar + b_e)
        f = torch.relu(self.encoder(x_bar))
        
        # Decoder: x_hat = W_d * f + b_d
        x_hat = self.decoder(f)

        # # Enforce unit-norm: normalize x_hat to lie on the unit circle.
        # epsilon = 1e-8  # small constant to avoid division by zero
        # x_hat = x_hat / (x_hat.norm(p=2, dim=1, keepdim=True) + epsilon)

        return x_hat, f

    def normalize_decoder_weights(self)-> None:
        """ Normalise decoder weights

            Normalise decoder weights to project values along them later. These decoder weights describe learned feature
            vectors.
        """
        # Enforce unit norm for each column of the decoder weight matrix
        # self.decoder.weight shape: (input_dim, hidden_dim)
        with torch.no_grad():
            weight = self.decoder.weight  # shape: (input_dim, hidden_dim)
            # Compute L2 norm over rows for each column (dim=0) and divide each column by its norm.
            norm = weight.norm(p=2, dim=0, keepdim=True) + 1e-8  # Avoid division by zero
            self.decoder.weight.copy_(weight / norm)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """ Encode data

            Subtracts the bias and feeds the data through the encoder layer. Returns a list of shape 
            (x.shape[0], hidden_dim). Meaning all data will be kept, but the size of the embeddings gets altered to the
            size of the number of hidden neurons.
            
            @param x: Input tensor.
            @type  x: torch.Tensor

            @rtype: torch.Tensor
            @return: Encoded data.
        """
        # Subtract decoder bias before encoding
        x_bar = x - self.decoder.bias
        # Compute the encoder output with ReLU non-linearity
        f = torch.relu(self.encoder(x_bar))
        return f
