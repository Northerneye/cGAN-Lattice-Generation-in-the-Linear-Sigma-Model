cGAN_pions_*_8x16... files are generated lattice configurations by the generator_epoch_28.pt
hmc_pions_traj_*... are dataset files
discriminator_epoch_#.pt and generator_epoch_#.pt are the saved networks
CSE_5835_Final_Project_Data.xlsx are the generated results

the commands:
    python cGAN_pytorch_Lattice.py    (will train the conditional cGAN)
    python cGAN_pytorch_Lattice.py --run 1   (will generate lattice configurations for alpha=0.0015(test set))
    python data_analysis_cGAN.py        (will run the data analysis and output the result in accelerated_analysis_results.csv)



Installation and Run Guide(through anaconda):
conda install pytorch torchvision torchaudio cpuonly -c pytorch
conda install numpy
conda install scipy

#cd into this folder
python cGAN_pytorch_Lattice.py

python cGAN_pytorch_Lattice.py --run 34
#34 is for the epoch number, insert the epoch number of interest

python data_analysis_cGAN.py
#output data is stored in accelerated_analysis_results.csv