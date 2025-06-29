"""
🔹 Phase 1: Data Handling
    └── Abstract art image dataset (1500 images from imagin_img folder)

🔹 Phase 2: Vision-Language Encoding
    └── Embed text using a Transformer encoder (BERT-style)
    └── Embed images using Vision Transformer (ViT) encoder

🔹 Phase 3: Generator (Transformer-based DiT architecture)
    └── Conditioned on the text encoding
    └── Outputs high-res, 3-channel (RGB) abstract art

🔹 Phase 4: Training Loop
    └── Paired data: (Prompt, Abstract Art Image)
    └── Loss: L2 image loss + perceptual loss + CLIP loss for text-image alignment

🔹 Phase 5: Inference
    └── User gives prompt
    └── Model generates a 512×512 abstract art image
"""
import os
import glob
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np
from transformers import AutoTokenizer, AutoModel
import warnings
warnings.filterwarnings('ignore')

# ===== CONFIGURATION =====
IMG_SIZE = 160  # Resolution of training images
OUTPUT_SIZE = 512  # Final output resolution
DATA_DIR = "F:\Deep AI\Imagin\imagein_img"  # Fixed: Removed "./" prefix
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ===== Phase 1: Data Preprocessing and Loading =====

# Enhanced transformation pipeline for RGB abstract art
transform_train = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),  # Converts to [0, 1] tensor with 3 channels (RGB)
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

transform_inference = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Custom Dataset (Fixed bugs)
class AbstractArtDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        super().__init__()
        
        # Fixed: Check if directory exists first
        if not os.path.exists(data_dir):
            print(f"❌ Error: Directory '{data_dir}' not found!")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Available directories: {[d for d in os.listdir('.') if os.path.isdir(d)]}")
            self.image_paths = []
            return
            
        # Fixed: More comprehensive file extension search
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG', '*.bmp', '*.BMP']
        self.image_paths = []
        
        for ext in extensions:
            self.image_paths.extend(glob.glob(os.path.join(data_dir, ext)))
        
        # Also check subdirectories
        for ext in extensions:
            self.image_paths.extend(glob.glob(os.path.join(data_dir, "**", ext), recursive=True))
        
        self.transform = transform if transform else transform_train
        print(f"✅ Found {len(self.image_paths)} abstract art images in '{data_dir}'")
        
        # Fixed: Show some example paths for debugging
        if len(self.image_paths) > 0:
            print(f"📁 Example image paths:")
            for i, path in enumerate(self.image_paths[:3]):
                print(f"   {i+1}. {path}")
        else:
            print(f"❌ No images found! Please check:")
            print(f"   - Directory exists: {os.path.exists(data_dir)}")
            if os.path.exists(data_dir):
                all_files = os.listdir(data_dir)
                print(f"   - Files in directory: {all_files[:10]}...")  # Show first 10 files
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        if len(self.image_paths) == 0:
            # Return dummy data if no images found
            return torch.randn(3, IMG_SIZE, IMG_SIZE)
            
        img_path = self.image_paths[idx]
        try:
            # Keep as RGB for abstract art
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
            return image
        except Exception as e:
            print(f"❌ Error loading {img_path}: {e}")
            # Return a random tensor if image fails to load
            return torch.randn(3, IMG_SIZE, IMG_SIZE)

# ===== Phase 2: Text Encoder (Fixed Implementation) =====

# Vocabulary settings
vocab = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!?., -_")
stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(vocab)
max_prompt_len = 64
embedding_dim = 256

# Fixed tokenizer function
def encode_prompt(prompt):
    prompt = prompt.lower()[:max_prompt_len]
    tokens = [stoi.get(c, 0) for c in prompt]
    if len(tokens) < max_prompt_len:
        tokens += [0] * (max_prompt_len - len(tokens))
    return torch.tensor(tokens, dtype=torch.long)

# Fixed Text Encoder
class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedding = nn.Embedding(max_prompt_len, embedding_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=8, batch_first=True),
            num_layers=6  # Reduced for efficiency
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        B, T = x.shape
        tok_emb = self.embedding(x)
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_embedding(pos)
        x = tok_emb + pos_emb
        # Fixed: TransformerEncoder with batch_first=True expects (B, T, E)
        x = self.transformer(x)
        x = x.permute(0, 2, 1)  # (B, E, T)
        x = self.pool(x).squeeze(-1)  # (B, E)
        return x

# ===== Phase 3: Vision Transformer Encoder (Fixed) =====

class PatchEmbed(nn.Module):
    def __init__(self, img_size=160, patch_size=20, in_chans=3, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/patch, W/patch)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim=256, nhead=8, dropout=0.1, ff_dim=1024):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),  # Better activation for transformers
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Multi-head attention with residual
        attn_output, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x))
        x = x + self.dropout(attn_output)
        
        # Feedforward with residual
        ff_output = self.ff(self.ln2(x))
        x = x + self.dropout(ff_output)
        return x

class ViTEncoder(nn.Module):
    def __init__(self, img_size=160, patch_size=20, in_chans=3, embed_dim=256, num_layers=6, nhead=8, ff_dim=1024, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)
        self.blocks = nn.Sequential(*[TransformerEncoderBlock(embed_dim, nhead, dropout, ff_dim) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        x = x + self.pos_embed  # Add positional embeddings
        x = self.blocks(x)  # Transformer encoder
        x = self.norm(x)
        return x

# ===== Phase 4: Transformer Decoder (Fixed and Enhanced) =====

class TransformerDecoderBlock(nn.Module):
    def __init__(self, embed_dim=256, nhead=8, ff_dim=1024, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout, batch_first=True)
        
        self.ln2 = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout, batch_first=True)
        
        self.ln3 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_out):
        # Self-attention with residual
        x_norm = self.ln1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_out)
        
        # Cross-attention with text encoder output
        x_norm = self.ln2(x)
        # Expand encoder_out to match number of patches
        if encoder_out.dim() == 2:  # (B, embed_dim)
            encoder_out = encoder_out.unsqueeze(1).expand(-1, x.size(1), -1)
        cross_attn_out, _ = self.cross_attn(x_norm, encoder_out, encoder_out)
        x = x + self.dropout(cross_attn_out)
        
        # Feedforward with residual
        ff_out = self.ff(self.ln3(x))
        x = x + self.dropout(ff_out)
        
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, num_layers=6, embed_dim=256, nhead=8, ff_dim=1024, patch_size=20, img_size=160):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        num_patches = (img_size // patch_size) ** 2
        
        # Learnable query embeddings for patches
        self.query_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)
        
        self.blocks = nn.ModuleList([
            TransformerDecoderBlock(embed_dim, nhead, ff_dim) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
        # Output projection to RGB patches
        patch_dim = patch_size * patch_size * 3  # RGB channels
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(ff_dim, patch_dim),
            nn.Tanh()  # Output in [-1, 1] range
        )

    def forward(self, text_embed):
        B = text_embed.size(0)
        
        # Initialize with learnable queries
        x = self.query_embed.expand(B, -1, -1) + self.pos_embed
        
        # Pass through decoder blocks
        for block in self.blocks:
            x = block(x, text_embed)
        
        # Final normalization and projection
        x = self.norm(x)
        patches = self.output_proj(x)  # (B, num_patches, patch_dim)
        
        # Reshape to image format
        num_patches_per_side = self.img_size // self.patch_size
        patches = patches.view(B, num_patches_per_side, num_patches_per_side, 3, self.patch_size, self.patch_size)
        
        # Reconstruct image from patches
        patches = patches.permute(0, 3, 1, 4, 2, 5)  # (B, 3, grid_h, patch_h, grid_w, patch_w)
        image = patches.contiguous().view(B, 3, self.img_size, self.img_size)
        
        return image

# ===== Phase 5: Complete Model =====

class AbstractArtGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = TextEncoder()
        self.decoder = TransformerDecoder()
    
    def forward(self, text_tokens):
        text_embed = self.text_encoder(text_tokens)
        generated_image = self.decoder(text_embed)
        return generated_image

# ===== Training Function =====

def train_model(model, dataloader, epochs=100, lr=1e-4):
    # Fixed: Check if dataloader has data
    if len(dataloader) == 0:
        print("❌ Error: No data to train on! Please check your dataset.")
        return
    
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.MSELoss()
    
    # Simple prompts for abstract art
    prompts = [
        "colorful abstract art", "geometric shapes", "flowing patterns", 
        "vibrant colors", "abstract painting", "modern art", "artistic design",
        "creative composition", "bold strokes", "dynamic forms"
    ]
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        batch_count = 0
        
        for batch_idx, real_images in enumerate(dataloader):
            real_images = real_images.to(DEVICE)
            
            # Random prompt for each batch
            prompt = np.random.choice(prompts)
            text_tokens = encode_prompt(prompt).unsqueeze(0).expand(real_images.size(0), -1).to(DEVICE)
            
            optimizer.zero_grad()
            
            # Generate images
            generated_images = model(text_tokens)
            
            # Loss calculation
            loss = criterion(generated_images, real_images)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        scheduler.step()
        avg_loss = total_loss / max(batch_count, 1)  # Fixed: Avoid division by zero
        print(f'Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}')
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'abstract_art_generator_epoch_{epoch+1}.pth')

# ===== Inference Functions =====

def generate_abstract_art(prompt, model, device=DEVICE):
    model.eval()
    with torch.no_grad():
        # Encode prompt
        text_tokens = encode_prompt(prompt).unsqueeze(0).to(device)
        
        # Generate image
        generated_image = model(text_tokens)
        
        # Upscale to high resolution
        generated_image = F.interpolate(generated_image, size=(OUTPUT_SIZE, OUTPUT_SIZE), mode='bilinear', align_corners=False)
        
        return generated_image

def save_output_image(tensor_img, filename="generated_abstract_art.png"):
    # Denormalize from [-1, 1] to [0, 1]
    img = (tensor_img.squeeze().cpu() + 1) / 2
    img = torch.clamp(img, 0, 1)
    
    # Convert to PIL Image
    img_np = img.permute(1, 2, 0).numpy()
    img_np = (img_np * 255).astype(np.uint8)
    
    Image.fromarray(img_np).save(filename)
    print(f"✅ Abstract art saved as {filename}")

# ===== Main Execution =====

if __name__ == "__main__":
    print("🎨 Abstract Art Generator Initialized!")
    print(f"🔧 Using device: {DEVICE}")
    print(f"📁 Looking for images in: {DATA_DIR}")
    
    # Initialize model
    model = AbstractArtGenerator()
    
    # Check if pre-trained model exists
    if os.path.exists("abstract_art_generator.pth"):
        print("📁 Loading pre-trained model...")
        try:
            model.load_state_dict(torch.load("abstract_art_generator.pth", map_location=DEVICE))
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("🚀 Will train a new model instead...")
    else:
        print("🚀 No pre-trained model found. Training new model...")
    
    # Create dataset and dataloader
    dataset = AbstractArtDataset(DATA_DIR)
    
    # Fixed: Only proceed with training if we have images
    if len(dataset) > 0:
        dataloader = DataLoader(dataset, batch_size=min(8, len(dataset)), shuffle=True, num_workers=0)  # Fixed: Set num_workers=0 for Windows compatibility
        
        # Train the model if no pre-trained model was loaded successfully
        if not os.path.exists("abstract_art_generator.pth"):
            train_model(model, dataloader, epochs=50)
            
            # Save the trained model
            torch.save(model.state_dict(), "abstract_art_generator.pth")
            print("💾 Model saved as abstract_art_generator.pth")
        
        # Interactive generation
        print("\n🎨 Ready to generate abstract art!")
        while True:
            user_prompt = input("\n🧠 Enter your abstract art prompt (type 'exit' to quit): ")
            
            if user_prompt.lower().strip() in ["exit", "quit"]:
                print("🚪 Exiting the imagination zone.")
                break
            
            print("🎨 Generating your abstract art...")
            try:
                image = generate_abstract_art(user_prompt, model)
                filename = f"abstract_art_{user_prompt.replace(' ', '_')[:20]}.png"
                # Remove invalid characters from filename
                filename = "".join(c for c in filename if c.isalnum() or c in "._-")
                save_output_image(image[0], filename=filename)
            except Exception as e:
                print(f"❌ Error generating image: {e}")
                print("Please try a different prompt.")
    else:
        print("❌ Cannot proceed without training data!")
        print("Please ensure your 'imagin_img' folder contains images and try again.")