import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import math
# --- NEW: Import for Mixed Precision ---
from torch.cuda.amp import GradScaler, autocast

# --- Configuration ---
IMG_DIR = "screenshots"  # Directory containing training images
MODEL_PATH = "diffusion_model.pth" # Path to save/load the model
OUTPUT_DIR = "generated_images" # Directory to save generated images
IMG_ORIG_WIDTH = 1920
IMG_ORIG_HEIGHT = 1080
# --- IMPORTANT: Resizing images for practical training ---
# Training on 1920x1080 is extremely demanding. We resize to a smaller size.
# Maintain aspect ratio roughly (1920/1080 = 16/9). Let's use 128x72.
IMG_SIZE_W = 128
IMG_SIZE_H = 72
IMG_CHANNELS = 3 # Assuming RGB images
BATCH_SIZE = 4     # Adjust based on GPU memory (can potentially increase with AMP)
# --- WARNING: 10000 epochs is very long. Consider starting lower (e.g., 500-1000) ---
NUM_EPOCHS = 500 # As requested, but monitor progress and consider stopping early.
LEARNING_RATE = 1e-4 # Initial learning rate
T_STEPS = 500      # Number of diffusion timesteps (denoising steps)
BETA_START = 1e-4
BETA_END = 0.02
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# --- Set num_workers based on OS ---
NUM_WORKERS = 2 # Keep at 2 unless you have specific reasons/hardware to change it

# --- NEW: Enable Mixed Precision only if using CUDA ---
USE_AMP = torch.cuda.is_available()

print(f"Using device: {DEVICE}")
print(f"Using Mixed Precision (AMP): {USE_AMP}")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
if not os.path.exists(IMG_DIR):
    print(f"Error: Image directory '{IMG_DIR}' not found.")
    print("Please create the directory and place your training images inside.")
    exit()

# --- Helper Function for Normalization (Replaces Lambda) ---
# Define the normalization function at the top level so it can be pickled.
def normalize_tensor(tensor):
    """Normalizes a tensor from [0, 1] to [-1, 1]."""
    return (tensor * 2) - 1

# --- Diffusion Utilities ---

# Create a linear beta schedule
betas = torch.linspace(BETA_START, BETA_END, T_STEPS, device=DEVICE)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
# Correct calculation for alphas_cumprod_prev using torch.cat
alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=DEVICE), alphas_cumprod[:-1]])

# Helper calculations for forward diffusion q(x_t | x_0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# Helper calculations for reverse diffusion q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
# Clamp variance to avoid potential division by zero or instability
# Ensure clamping happens on the tensor directly
posterior_variance_clipped = torch.clamp(posterior_variance, min=1e-20)
posterior_log_variance_clipped = torch.log(posterior_variance_clipped)


posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)

# Function to extract the correct alpha/beta values for a given timestep t and batch size
def extract(a, t, x_shape):
    batch_size = t.shape[0]
    # Ensure 'a' is on the same device as 't' before gather
    a = a.to(t.device)
    out = a.gather(-1, t)
    # Reshape to match the image dimensions for broadcasting
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

# Forward diffusion process: Add noise to an image
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)

    noisy_image = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    return noisy_image

# --- Dataset ---
class ImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        # List comprehension to get image file paths
        self.img_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir)
                          if os.path.isfile(os.path.join(img_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not self.img_files:
            print(f"Warning: No images found in directory '{img_dir}'. Training might fail or produce poor results.")


    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        try:
            # Open image and ensure it's RGB
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            # Ensure the returned tensor has the correct shape
            if image.shape[1:] != (IMG_SIZE_H, IMG_SIZE_W):
                 # This case should ideally not happen if transforms are correct
                 print(f"Warning: Image {img_path} has unexpected shape {image.shape} after transform. Expected H={IMG_SIZE_H}, W={IMG_SIZE_W}. Skipping.")
                 # Return a placeholder tensor of the correct shape
                 return torch.zeros((IMG_CHANNELS, IMG_SIZE_H, IMG_SIZE_W))
            return image
        except Exception as e:
            print(f"Warning: Could not load or process image {img_path}: {e}. Skipping.")
            # Return a placeholder tensor of the correct shape if loading fails
            return torch.zeros((IMG_CHANNELS, IMG_SIZE_H, IMG_SIZE_W))


# Define transformations: Resize, convert to tensor, normalize using the defined function
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE_H, IMG_SIZE_W)), # Resize images
    transforms.ToTensor(),              # Convert PIL image to Tensor (range [0, 1])
    transforms.Lambda(normalize_tensor) # Use the defined function here
])

dataset = ImageDataset(IMG_DIR, transform=transform)

# Simple check if the dataset is empty after initialization
if len(dataset) == 0:
     print(f"Error: No valid images could be processed in {IMG_DIR}. Exiting.")
     exit()

# Custom collate function to filter out problematic data (e.g., tensors with wrong shape)
def collate_fn(batch):
    # Filter out items that are not tensors or have incorrect shape (e.g., from failed loads)
    batch = [item for item in batch if isinstance(item, torch.Tensor) and item.shape == (IMG_CHANNELS, IMG_SIZE_H, IMG_SIZE_W)]
    if not batch:
        return None # Return None if the batch becomes empty after filtering
    return torch.stack(batch)


dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS, # Use the configured number of workers
    drop_last=True,          # Drop last incomplete batch
    pin_memory=True if DEVICE == "cuda" else False, # Pin memory if using GPU
    collate_fn=collate_fn    # Use custom collate function
)


# --- Model Architecture (Simple U-Net) ---
# Basic building block for U-Net
class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        # Time embedding projection
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            # Upsampling block
            # First conv takes 2*in_ch because we concatenate skip connection in U-Net forward pass
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1) # Upsample H, W by 2
        else:
            # Downsampling block
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1) # Downsample H, W by 2
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t_emb):
        # x: input features (potentially concatenated in U-Net forward)
        # t_emb: time embedding

        # First Conv -> BN -> ReLU
        h = self.bnorm1(self.relu(self.conv1(x)))

        # Project time embedding and add to features
        time_emb_proj = self.relu(self.time_mlp(t_emb))
        # Reshape time embedding to be added channel-wise
        time_emb_proj = time_emb_proj.unsqueeze(-1).unsqueeze(-1) # Shape: (B, out_ch, 1, 1)
        h = h + time_emb_proj # Add time embedding

        # Second Conv -> BN -> ReLU
        h = self.bnorm2(self.relu(self.conv2(h)))

        # Apply Upsampling or Downsampling
        return self.transform(h)


# Sinusoidal time embedding
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # Ensure dim is even
        if self.dim % 2 != 0:
            raise ValueError("SinusoidalPositionEmbeddings requires an even dimension.")


    def forward(self, time):
        # time: Tensor of shape (batch_size,) representing timesteps
        device = time.device
        half_dim = self.dim // 2
        # Calculate frequency embeddings
        embeddings = math.log(10000) / (half_dim - 1)
        # Create tensor of frequencies: [0, 1, ..., half_dim-1]
        freqs = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        # Calculate arguments for sin and cos: time[:, None] * freqs[None, :]
        # Shape: (batch_size, 1) * (1, half_dim) -> (batch_size, half_dim)
        args = time[:, None] * freqs[None, :]
        # Concatenate sin and cos embeddings
        embeddings = torch.cat((args.sin(), args.cos()), dim=-1) # Shape: (batch_size, dim)
        return embeddings

# The U-Net Model
class SimpleUNet(nn.Module):
    """
    A simplified U-Net architecture for the diffusion model.
    Includes skip connections and time embeddings.
    """
    def __init__(self, img_channels=3, time_emb_dim=256, base_dim=64):
        super().__init__()
        # Ensure time_emb_dim is provided
        if time_emb_dim is None:
            raise ValueError("time_emb_dim must be specified for SimpleUNet.")

        # Time embedding layer
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        # Initial projection layer (maps input channels to base_dim)
        self.init_conv = nn.Conv2d(img_channels, base_dim, 3, padding=1) # -> 64 channels

        # Downsampling path
        self.down1 = Block(base_dim, base_dim*2, time_emb_dim)      # 64 -> 128
        self.down2 = Block(base_dim*2, base_dim*4, time_emb_dim)    # 128 -> 256
        self.down3 = Block(base_dim*4, base_dim*4, time_emb_dim)    # 256 -> 256

        # Bottleneck - adjusted channels
        self.bot1 = nn.Conv2d(base_dim*4, base_dim*8, 3, padding=1) # 256 -> 512
        self.bot_relu1 = nn.ReLU()
        self.bot2 = nn.Conv2d(base_dim*8, base_dim*8, 3, padding=1) # 512 -> 512
        self.bot_relu2 = nn.ReLU()
        self.bot3 = nn.Conv2d(base_dim*8, base_dim*4, 3, padding=1) # 512 -> 256
        self.bot_relu3 = nn.ReLU()


        # --- Corrected Upsampling Path ---
        self.up1_corrected = Block(in_ch=base_dim*4, out_ch=base_dim*4, time_emb_dim=time_emb_dim, up=True) # In: 256 -> Out: 256
        self.up2_corrected = Block(in_ch=base_dim*4, out_ch=base_dim*2, time_emb_dim=time_emb_dim, up=True) # In: 256 -> Out: 128
        self.up3_corrected = Block(in_ch=base_dim*2, out_ch=base_dim, time_emb_dim=time_emb_dim, up=True)   # In: 128 -> Out: 64


        # Final output layer
        # Input channels: output of last up block (u3[64]) + initial conv features skip (x_init[64]) = 128
        self.out_conv = nn.Conv2d(base_dim*2, img_channels, 1) # Input 128, Output img_channels

    def forward(self, x, t):
        # x: Input image batch (B, C, H, W)
        # t: Timestep batch (B,)

        # Get time embeddings
        t_emb = self.time_mlp(t) # (B, time_emb_dim)

        # Initial convolution
        x_init = self.init_conv(x) # (B, base_dim, H, W) -> (B, 64, H, W)

        # --- Downsampling ---
        d1 = self.down1(x_init, t_emb) # (B, 128, H/2, W/2)
        d2 = self.down2(d1, t_emb)     # (B, 256, H/4, W/4)
        d3 = self.down3(d2, t_emb)     # (B, 256, H/8, W/8)


        # --- Bottleneck ---
        b = self.bot_relu1(self.bot1(d3)) # (B, 512, H/8, W/8)
        b = self.bot_relu2(self.bot2(b)) # (B, 512, H/8, W/8)
        b = self.bot_relu3(self.bot3(b)) # (B, 256, H/8, W/8)


        # --- Upsampling with skip connections ---
        u1 = self.up1_corrected(torch.cat((b, d3), dim=1), t_emb)   # (B, 256, H/4, W/4)
        u2 = self.up2_corrected(torch.cat((u1, d2), dim=1), t_emb)  # (B, 128, H/2, W/2)
        u3 = self.up3_corrected(torch.cat((u2, d1), dim=1), t_emb) # (B, 64, H, W)


        # Final convolution
        output = self.out_conv(torch.cat((u3, x_init), dim=1)) # (B, img_channels, H, W)

        return output

# --- Loss Function ---
loss_fn = nn.MSELoss() # Predict the noise

# --- Model, Optimizer ---
model = SimpleUNet(img_channels=IMG_CHANNELS, time_emb_dim=256).to(DEVICE) # Ensure time_emb_dim is passed
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- NEW: Learning Rate Scheduler ---
# Cosine Annealing decays the learning rate following a cosine curve.
# T_max is the number of iterations until the first restart (here, total epochs).
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6) # eta_min is the minimum LR

# --- NEW: Gradient Scaler for Mixed Precision ---
# Creates a GradScaler instance for managing gradient scaling with AMP
# Enabled only if USE_AMP is True
scaler = GradScaler(enabled=USE_AMP)

# --- Training Loop ---
def train_model():
    print("Starting training...")
    if len(dataloader) == 0:
        print("Error: DataLoader is empty. Check image loading and filtering.")
        return # Exit training if dataloader is empty

    # --- Determine total steps for scheduler if needed per-step ---
    # total_steps = len(dataloader) * NUM_EPOCHS # If scheduler needs step updates

    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        # Wrap dataloader with tqdm for progress bar
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=True) # Keep bar after epoch finishes

        for step, batch in enumerate(progress_bar):
            # Check if batch is None (due to collate_fn filtering)
            if batch is None:
                 continue
            # Ensure batch contains valid tensors (double check after collate)
            if not isinstance(batch, torch.Tensor) or batch.nelement() == 0:
                 print(f"Warning: Skipping invalid batch content at step {step}.")
                 continue

            # --- UPDATED: Use set_to_none=True for potential minor speedup ---
            optimizer.zero_grad(set_to_none=True) # Reset gradients

            batch = batch.to(DEVICE) # Move batch to target device

            # Sample random timesteps t for each image in the batch
            current_batch_size = batch.shape[0]
            t = torch.randint(0, T_STEPS, (current_batch_size,), device=DEVICE).long()

            # Sample noise and create noisy images (forward process)
            noise = torch.randn_like(batch)
            x_noisy = q_sample(x_start=batch, t=t, noise=noise)

            # --- NEW: Autocast context manager for Mixed Precision ---
            # Runs the forward pass under autocast, using float16 where appropriate
            with autocast(enabled=USE_AMP):
                predicted_noise = model(x_noisy, t)
                loss = loss_fn(predicted_noise, noise) # Calculate loss

            # --- NEW: Scale loss and backpropagate using GradScaler ---
            # scaler.scale scales the loss, scaler.step unscales gradients and updates optimizer
            # scaler.update prepares for the next iteration
            scaler.scale(loss).backward() # Scales loss. Calls backward() on scaled loss
            scaler.step(optimizer)        # Unscales gradients and calls optimizer.step()
            scaler.update()               # Updates the scale for next iteration

            epoch_loss += loss.item() # Accumulate loss for the epoch

            # Update progress bar postfix with current loss and learning rate
            current_lr = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.6f}")

        # --- NEW: Step the scheduler after each epoch ---
        scheduler.step()

        # Calculate average loss for the epoch
        avg_epoch_loss = epoch_loss / len(dataloader) if len(dataloader) > 0 else 0
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} finished. Average Loss: {avg_epoch_loss:.4f}")

        # Optional: Save model periodically (e.g., every 100 epochs)
        if (epoch + 1) % 100 == 0 or epoch == NUM_EPOCHS - 1: # Save on milestones and at the end
             checkpoint_path = f"{MODEL_PATH}_epoch_{epoch+1}.pth"
             print(f"Saving model checkpoint to {checkpoint_path}...")
             try:
                 # Save model state dict directly
                 torch.save(model.state_dict(), checkpoint_path)
             except Exception as e:
                 print(f"Error saving checkpoint: {e}")


    print("Training finished.")
    # Save the final model
    print(f"Saving final model to {MODEL_PATH}...")
    try:
        torch.save(model.state_dict(), MODEL_PATH)
    except Exception as e:
        print(f"Error saving final model: {e}")


# --- Sampling/Generation Function ---
@torch.no_grad() # Disable gradient calculations for inference
def sample(model, n_images=1):
    print(f"Generating {n_images} image(s)...")
    model.eval() # Set model to evaluation mode (disables dropout, etc.)

    # Start with random noise (initial state in the reverse process)
    img_shape = (n_images, IMG_CHANNELS, IMG_SIZE_H, IMG_SIZE_W)
    img = torch.randn(img_shape, device=DEVICE)

    # Denoising loop (from T_STEPS-1 down to 0)
    # Wrap the loop with tqdm for progress visualization
    for i in tqdm(reversed(range(0, T_STEPS)), desc="Sampling", total=T_STEPS, leave=False):
        # Create a tensor of the current timestep 'i' for the batch
        t = torch.full((n_images,), i, device=DEVICE, dtype=torch.long)

        # --- UPDATED: Use autocast during inference if AMP was used for training ---
        # Although not strictly necessary for correctness (gradients aren't computed),
        # using autocast can sometimes speed up inference slightly on compatible hardware.
        with autocast(enabled=USE_AMP):
            predicted_noise = model(img, t)

        # Get diffusion parameters for the current timestep t
        alpha_t = extract(alphas, t, img.shape)
        alpha_t_cumprod = extract(alphas_cumprod, t, img.shape)
        beta_t = extract(betas, t, img.shape)
        sqrt_one_minus_alpha_t_cumprod = extract(sqrt_one_minus_alphas_cumprod, t, img.shape)
        sqrt_recip_alpha_t = torch.sqrt(1.0 / alpha_t) # 1 / sqrt(alpha_t)

        # Calculate the mean of the distribution p(x_{t-1} | x_t)
        model_mean = sqrt_recip_alpha_t * (img - beta_t * predicted_noise / sqrt_one_minus_alpha_t_cumprod)

        if i == 0:
            # Last step (t=0), the output is the final denoised image (mean)
            img = model_mean
        else:
            # Add noise (variance) for intermediate steps
            posterior_log_variance_t = extract(posterior_log_variance_clipped, t, img.shape)
            posterior_variance_t = torch.exp(posterior_log_variance_t)
            noise = torch.randn_like(img)
            # Algorithm 2 line 4: x_{t-1} = mean + sqrt(variance) * z (noise)
            img = model_mean + torch.sqrt(posterior_variance_t) * noise

    model.train() # Set model back to training mode

    # Post-process image: scale back from [-1, 1] to [0, 1] and clamp
    img = (img + 1) * 0.5
    img = torch.clamp(img, 0.0, 1.0)
    return img


# --- Main Execution ---
if __name__ == "__main__":
    # Check if a trained model exists
    model_ready = False
    if os.path.exists(MODEL_PATH):
        print(f"Found existing model at {MODEL_PATH}. Loading weights...")
        try:
             # Load state dict, ensuring map_location handles CPU/GPU transfer
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            print("Model loaded successfully.")
            # Set flag to skip training if model is loaded
            generate_only = True
            model_ready = True # Model is ready if loaded successfully
        except Exception as e:
            print(f"Error loading model: {e}. Model file might be corrupted or incompatible.")
            print("Proceeding to train from scratch.")
            generate_only = False
            # Training will happen below
    else:
        print("No existing model found. Training from scratch...")
        generate_only = False
        # Training will happen below

    # Train the model if it wasn't loaded successfully or doesn't exist
    if not generate_only:
        train_model()
        # Check if training produced a model file, indicating it might be ready
        if os.path.exists(MODEL_PATH):
             model_ready = True
        else:
             print("Training finished, but model file was not saved. Cannot generate image.")


    # Generate an image only if the model is ready (loaded or successfully trained)
    if model_ready:
        # Ensure model weights are loaded if we skipped training but the file exists
        # (This check might be redundant given the logic above, but adds safety)
        if generate_only and not hasattr(model, 'up1_corrected'): # Simple check if model looks initialized
             print("Model structure seems missing despite file existing, attempting to load again...")
             try:
                 model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
                 print("Model re-loaded successfully.")
             except Exception as e:
                 print(f"Failed to reload model: {e}. Cannot generate image.")
                 model_ready = False # Prevent generation attempt

        if model_ready:
            generated_image = sample(model, n_images=1)
            # Save the generated image
            output_path = os.path.join(OUTPUT_DIR, "generated_image.png")
            try:
                save_image(generated_image[0], output_path) # Save the first image in the batch
                print(f"Generated image saved to {output_path}")
                print(f"Note: Image resolution is {IMG_SIZE_W}x{IMG_SIZE_H} due to training constraints.")
                print(f"Original requested size was {IMG_ORIG_WIDTH}x{IMG_ORIG_HEIGHT}.")
            except Exception as e:
                print(f"Error saving generated image: {e}")

    elif generate_only:
         # This case means loading failed initially
         print("Model loading failed. Cannot generate image.")
    else: # This case implies training was initiated but failed before creating MODEL_PATH
        print("Model training did not complete successfully or was skipped, cannot generate image.")

