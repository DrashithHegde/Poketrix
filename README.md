# Poketrix: Conditional GAN for Pokémon Generation and Detection

**Poketrix** is a full-stack Artificial Intelligence application developed as a coursework assignment. It features a custom-built **Conditional Generative Adversarial Network (cGAN)** that can synthesize novel 64x64 Pokémon images conditioned on specific characteristics (Type and Color) and classify user-uploaded images as real or AI-generated.

## AI Architecture (cGAN)
This project implements a Deep Convolutional Conditional GAN framework using PyTorch:
* **The Generator** uses transposed convolutional layers (`ConvTranspose2d`) to map a 100-dimensional latent noise vector and a 10-dimensional condition vector into a 64x64x3 RGB image space. 
* **The Discriminator** uses standard convolutional layers (`Conv2d`) with LeakyReLU activations to downsample the image and condition matrix, calculating a Binary Cross-Entropy (BCE) loss to distinguish between authentic Kaggle Pokémon sprites and synthesized fakes.
* **Adversarial Training:** The two neural networks undergo adversarial training via deep matrix multiplication (with full CUDA/GPU acceleration support), competing mathematically until the Generator learns the underlying data distribution of the traits (e.g., Red + Fire type).

## Technology Stack
* **AI / Deep Learning:** PyTorch, Torchvision, CUDA (Hardware Acceleration)
* **Data Processing:** Pandas, NumPy, Pillow (PIL)
* **Backend API:** FastAPI, Uvicorn, Python-Multipart
* **Frontend UI:** React (Vite), Axios, HTML/CSS

## Project Structure
```text
Poketrix/
├── backend/               # Deep learning models and FastAPI server
│   ├── data/              # Kaggle Pokemon dataset (images & pokedex.csv)
│   ├── models/            # Neural network architectures (generator.py, discriminator.py)
│   ├── utils/             # Preprocessing logic & data loaders
│   ├── train.py           # The PyTorch adversarial training loop
│   └── main.py            # FastAPI endpoints (/generate, /detect)
└── frontend/              # The React Pokédex User Interface
    ├── src/               # React components (GenerateMode.jsx, DetectMode.jsx)
    └── package.json       # Node.js dependencies
```

## Setup and Installation

### 1. Backend & AI Training
Ensure you have Python 3.12+ installed. If you have an NVIDIA GPU, installing the CUDA-enabled version of PyTorch is highly recommended for faster training.
```bash
# Navigate to backend
cd backend

# Install dependencies
pip install torch torchvision pandas numpy pillow fastapi uvicorn python-multipart

# Execute the training loop (Trains for 50 Epochs)
python train.py

# Start the API Server
python main.py
```
*The server will run on `http://127.0.0.1:8000`*

### 2. Frontend UI
Ensure you have Node.js installed. Open a new terminal.
```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```
*The dashboard will compile and be accessible at `http://localhost:5173`*

## 🎮 Usage
Once both servers are running, access the web UI:
1. **Generate Mode:** Select a Pokémon Type (e.g., Fire, Water) and a Primary Color. The backend will sample a random noise vector, concatenate the condition vector, and perform inference through the Generator to create a new Pokémon. The Discriminator evaluates the result in real-time, providing a "Real vs. Fake" confidence probability.
2. **Detect Mode:** Upload an external image. The image is passed through the normalization preprocessing pipeline and fed to the Discriminator for binary classification.

## 📝 Academic Notes & Limitations
* **Resolution Constraint:** The model outputs `64x64` images. This constraint was purposefully chosen to maintain a rapid feedback loop and ensure the model could finish training locally on consumer hardware (RTX 4050) within a tight time frame. Super-resolution (e.g., `256x256`) would simply require expanding the convolutional block depth and increasing the epoch count.
* **Epochs:** Training is currently capped at 50 epochs to prevent local machine overheating while still clearly demonstrating functional adversarial convergence.
