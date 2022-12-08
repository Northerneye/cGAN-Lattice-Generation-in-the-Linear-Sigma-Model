FILES CONTAINED:
cGAN_pions_*_8x16... files are generated lattice configurations by the generator_epoch_28.pt
hmc_pions_traj_*... are dataset files
discriminator_epoch_#.pt and generator_epoch_#.pt are the saved networks
CSE_5835_Final_Project_Data.xlsx are the generated results

commands:
    python cGAN_pytorch_Lattice.py    (will train the conditional cGAN)
    python cGAN_pytorch_Lattice.py --run 1   (will generate lattice configurations for alpha=0.0015(test set))
    python data_analysis_cGAN.py        (will run the data analysis and output the result in accelerated_analysis_results.csv)



INSTALLATION AND RUNNING GUIDE(through anaconda):
1. conda install pytorch torchvision torchaudio cpuonly -c pytorch
2. conda install numpy
3. conda install scipy

cd into this directory
4. python cGAN_pytorch_Lattice.py

5. python cGAN_pytorch_Lattice.py --run 34
    NOTE: 34 is for the epoch number being investigated.  This command will generate lattice configurations from the epoch 34 model. Insert the epoch number of interest if a different epoch is desired

6. python data_analysis_cGAN.py
    NOTE: output data is stored in accelerated_analysis_results.csv
