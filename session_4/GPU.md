## Instructions to train on GPU

This tutorial will help you set up a GPU on the cloud to train a large model.

This tutorial uses [Lambda Labs](https://lambdalabs.com/). You should feel free to use other providers if they work for you.

Important Note: with Lambda Labs and many other GPU providers, you need to *terminate* your instance to stop paying for it (you cannot just shut it down). With providers like AWS, you can shut down your instance and stop paying for it, but you pay for this flexibility with higher prices.

1. Create an account on [Lambda Labs](https://lambdalabs.com/)
2. Launch an instance (e.g. A10)
3. During the first launch, create a filesystem and give it a name.
4. You can then reuse this filesystem later on when creating a new instance
5. Write down the IP address of your instance
6. Install the Remote SSH module in Visual Code
7. Create an SSH key on your local machine (e.g. on Mac/Linux: `ssh-keygen` then `cat ~/.ssh/id_rsa.pub`)
8. Add the SSH key to your Lambda Labs account
9. Launch a remote session in Visual Code `ssh ubuntu@<IP-address>`
10. Verify that NVIDIA drivers are installed: `nvidia-smi` should list the GPU on your machine
11. Open a ssh session on the GPU machine. Checkout your code there. Make sure to work in your filesystem (it should appear at the root of your home directory).
12. Run the diffusion trial program `diffusion-trial.py`. It should generate an image of an astronaut riding a horse.

### Other providers:
- AWS
- Cudo
- Paperspace
- Hyperstack
- Vast.ai
- Google Cloud / Colab
