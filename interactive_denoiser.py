import tkinter as tk
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tkinter import PhotoImage
from PIL import Image, ImageTk
from einops import rearrange
import torch.nn as nn

def init_model():
    class Autoencoder(nn.Module):
        def __init__(self, latent_dim):
            super(Autoencoder, self).__init__()
            self.latent_dim = latent_dim
            self.__setup_encoder()
            self.__setup_decoder()
        
        def __setup_encoder(self):
            self.enc_conv1 = nn.Conv2d(1, 512, kernel_size=3, stride=2, padding=1) # Output: 512 x 14 x 14
            self.enc_relu1 = nn.ReLU()
            self.enc_conv2 = nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1) # Output: 256 x 7 x 7
            self.enc_relu2 = nn.ReLU()
            self.enc_conv3 = nn.Conv2d(256, 128, kernel_size=7) # Output: 128 x 1 x 1
            self.enc_linear = nn.Linear(128, self.latent_dim) # Output: 1 x latent_dim

        def __setup_decoder(self):
            self.dec_linear = nn.Linear(self.latent_dim, 128) # Output: 1 x 128
            self.dec_conv1 = nn.ConvTranspose2d(128, 256, kernel_size=7) # Output: 512 x 7 x 7
            self.dec_relu1 = nn.ReLU()
            self.dec_conv2 = nn.ConvTranspose2d(256, 512, kernel_size=3, stride=2, padding=1, output_padding=1) # Output: 512 x 14 x 14
            self.dec_relu2 = nn.ReLU()
            self.dec_conv3 = nn.ConvTranspose2d(512, 1, kernel_size=3, stride=2, padding=1, output_padding=1) # Output: 1 x 28 x 28
            self.dec_tanh = nn.Tanh()
        
        def encode(self, x):
            assert x.shape[-3:] == (1, 28, 28)
            x = self.enc_conv1(x)
            x = self.enc_relu1(x)
            x = self.enc_conv2(x)
            x = self.enc_relu2(x)
            x = self.enc_conv3(x)
            x = rearrange(x, 'b c h w -> b (c h w)') # Remove h, w dimensions which are 1
            x = self.enc_linear(x)
            return x
        
        def decode(self, latent):
            assert latent.shape[-1] == self.latent_dim
            x = self.dec_linear(latent)
            x = rearrange(x, 'b c -> b c 1 1') # Add h, w dimensions which are 1, prepare to add spatial information
            x = self.dec_conv1(x)
            x = self.dec_relu1(x)
            x = self.dec_conv2(x)
            x = self.dec_relu2(x)
            x = self.dec_conv3(x)
            x = self.dec_tanh(x)
            return x

        def forward(self, x):
            latent = self.encode(x)
            x_reconstructed = self.decode(latent)
            return x_reconstructed
    
    return Autoencoder(latent_dim=128)

def add_gaussian_noise(image, mean=0.0, std=0.1):
    noise = torch.randn_like(image) * std + mean
    noisy_image = image + noise
    noisy_image = torch.clip(noisy_image, 0., 1.)
    return noisy_image

def convert_tensor_image_to_pil_image(image):
    # To numpy and scale to 0-255 (RGB)
    image = image.view(28, 28).numpy()
    image = image * 255

    # Convert to PIL image
    image = Image.fromarray(image)

    # Upsample the image to 280x280
    return image.resize((112, 112), Image.Resampling.NEAREST)

class TkinterApp:
    def __init__(self, root, model, mnist_loader):
        self.root = root
        self.model = model
        self.mnist_loader = mnist_loader

        # Center window
        self.root.eval('tk::PlaceWindow . center')

        self.root.title("Interactive Denoiser")
        self.root.geometry("400x300")

        # Create 3 columns with one label above each image: original, noisy, denoised
        self.__init_label_image_table()

        # Create noise controls
        self.__init_noise_controls()

    def __init_label_image_table(self):
        # Create first column

        # Create image label
        tk.Label(text="Original image").grid(row=0, column=0)

        # Load original image from dataset
        self.load_random_mnist_image() # Load PIL image into self.image
        self.draw_original_image()

        

        # Create second column
        tk.Label(text="Noisy image").grid(row=0, column=1)
        self.draw_noisy_image()

        # Create third column
        tk.Label(text="Denoised image").grid(row=0, column=2)
        self.draw_reconstructed_image()
    
    def __init_noise_controls(self):
        # New image button
        self.btn_load_random_image = tk.Button(self.root, text="New image from dataset", command=self.btn_load_random_img_clicked)
        self.btn_load_random_image.grid(row=2, column=0)

        tk.Label(text="Gaussian noise controls").grid(row=2, column=1)

        # Gaussian mean noise slider
        self.noise_mean = 0.0 # Default value
        self.mean_noise_slider = tk.Scale(self.root, from_=-0.5, to=0.5, resolution=0.01, orient=tk.HORIZONTAL, label="Mean", command=self.mean_noise_slider_moved)
        self.mean_noise_slider.grid(row=3, column=1)
        self.mean_noise_slider.set(self.noise_mean) # Set default value

        # Gaussian std noise slider
        self.noise_std = 0.25 # Default value
        self.std_noise_slider = tk.Scale(self.root, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, label="Variance", command=self.std_noise_slider_moved)
        self.std_noise_slider.grid(row=4, column=1)
        self.std_noise_slider.set(self.noise_std) # Set default value
    
    def redraw_image_with_noise(self):
        # Add Gaussian noise to the image
        self.noisy_image = add_gaussian_noise(self.original_image, self.noise_mean, self.noise_std)

        # Draw the noisy image
        self.draw_noisy_image()

    def mean_noise_slider_moved(self, value):
        self.noise_mean = float(value)
        self.redraw_image_with_noise()

        # Also redraw denoised image
        self.draw_reconstructed_image()

    def std_noise_slider_moved(self, value):
        self.noise_std = float(value)
        self.redraw_image_with_noise()

        # Also redraw denoised image
        self.draw_reconstructed_image()
    
    def btn_load_random_img_clicked(self):
        # Redraw original image
        self.load_random_mnist_image()
        self.draw_original_image()

        # Also redraw noisy image
        self.redraw_image_with_noise()

        # Redraw denoised image
        self.draw_reconstructed_image()
    
    def load_random_mnist_image(self):
        # Sample random number between 0 and len(mnist_loader)
        random_index = torch.randint(0, len(self.mnist_loader.dataset), (1,)).item()

        # Load image and label
        self.original_image, self.label = self.mnist_loader.dataset[random_index]

        # Noisy image is also a copy of the original image (with added noise later)
        self.noisy_image = self.original_image.clone()
    

    
    def draw_original_image(self):
        # Convert tensor image to PIL image
        pil_image = convert_tensor_image_to_pil_image(self.original_image)
        self.tk_image = ImageTk.PhotoImage(pil_image)

        # Create image label (show it)
        self.original_image_label = tk.Label(root, image = self.tk_image)
        self.original_image_label.image = self.tk_image
        self.original_image_label.grid(row=1, column=0)
    
    def draw_noisy_image(self):
        # Convert tensor image to PIL image
        pil_image = convert_tensor_image_to_pil_image(self.noisy_image)
        self.tk_noisy_image = ImageTk.PhotoImage(pil_image)

        # Create image label (show it)
        self.draw_noisy_image_image_label = tk.Label(root, image = self.tk_noisy_image)
        self.draw_noisy_image_image_label.image = self.tk_noisy_image
        self.draw_noisy_image_image_label.grid(row=1, column=1)

    def draw_reconstructed_image(self):
        # Pass noisy image through the model
        with torch.no_grad():
            self.reconstructed_image = self.model(self.noisy_image.unsqueeze(0)).squeeze(0)

        # Convert tensor image to PIL image
        pil_image = convert_tensor_image_to_pil_image(self.reconstructed_image)
        self.tk_reconstructed_image = ImageTk.PhotoImage(pil_image)

        # Create image label (show it)
        self.reconstructed_image_label = tk.Label(root, image = self.tk_reconstructed_image)
        self.reconstructed_image_label.image = self.tk_reconstructed_image
        self.reconstructed_image_label.grid(row=1, column=2)

    def run(self):
        self.root.mainloop()

def load_mnist_loader(batch_size=1):
    # Load MNIST dataset
    mnist_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
        ])),
        batch_size=batch_size, shuffle=True
    )
    return mnist_loader

# Check if the app is still running (CTRL+C interrupt)
def check():
    root.after(50, check)

if __name__ == "__main__":
    model = init_model()
    model.load_state_dict(torch.load("autoencoder_after_noise_training.pth"))
    model.eval() # Evaluation mode

    # Load MNIST dataset
    mnist_loader = load_mnist_loader()

    # Create the app
    root = tk.Tk()
    root.after(50, check)
    app = TkinterApp(root, model, mnist_loader)

    try:
        app.run()
    except KeyboardInterrupt:
        pass