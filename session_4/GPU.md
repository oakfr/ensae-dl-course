## Instructions to train on GPU

This tutorial will help you set up a GPU on the cloud to train a large model.

This tutorial uses [Lambda Labs](https://lambdalabs.com/). You should feel free to use other providers if they work for you.

Important Note: with Lambda Labs and many other GPU providers, you need to *terminate* your instance to stop paying for it (you cannot just shut it down). With providers like AWS, you can shut down your instance and stop paying for it, but you pay for this flexibility with higher prices.

1. Create an account on [Lambda Labs](https://lambdalabs.com/)
2. Launch an instance (e.g. A10)
3. During the first launch, create a filesystem and give it a name. You can then reuse this filesystem later on when creating a new instance.
4. Write down the IP address of your instance
5. Create an SSH key on your local machine (e.g. on Mac/Linux: `ssh-keygen` then `cat ~/.ssh/id_rsa.pub`)
6. Add the SSH key to your Lambda Labs account
7. Connect via SSH to your GPU machine and create another SSH key there (`ssh-keygen` then `cat ~/.ssh/id_rsa.pub`)
8. Add this SSH key to your github profile
9. On the GPU machine, clone this repo and go to the `session_4` directory
10. Verify that NVIDIA drivers are installed: `nvidia-smi` should show the GPU on your machine
11. Install the required python packages: `pip3 install -r requirements.txt`
12. Run the diffusion trial program `python3 diffusion-trial.py`. It should generate an image of an astronaut riding a horse.
13. Install the Remote SSH module in Visual Code
14. Launch a remote session in Visual Code `ssh ubuntu@<GPU-IP-address>`
15. Open the repo folder. You can now run code and notebooks remotely via Visual Code.
16. For step 15, you will likely need to install a jupyter/python environment on your new GPU machine. You can do this remotely via Visual Code by following its instructions.

Make sure to always work in your Lambda Labs filesystem (it should appear at the root of your home directory) otherwise all your data will be lost when you kill the instance.

** DO NOT FORGET TO TERMINATE YOUR GPU ONCE YOU ARE DONE WITH YOUR EXPERIMENT **

### Other GPU providers
- AWS
- Cudo
- Paperspace
- Hyperstack
- Vast.ai
- Google Cloud / Colab
