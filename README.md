# Learning-noise-induced-transitions-by-multi-scaling-reservoir-computing
Code for the paper "Learning noise-induced transitions by multi-scaling reservoir computing" (https://www.nature.com/articles/s41467-024-50905-w).

# Reservoir-Computer

A software package for the manuscript "Learning noise-induced transitions by multi-scaling reservoir computing".

Authors: Zequn Lin, Zhaofan Lu, Zengru Di, Ying Tang

# Running by choosing a code **Set as file to run**

The output number is the spectral radius of the corresponding system's reservoir connection.

## Code

In order to facilitate the distinction, we divided the code into 4 folders: 

(1) Main text: code for the examples in the main text, including **A bistable gradient system with white noise** , **A bistable gradient system with colored noise** , **A bistable non-gradient system** , and **Experimental data of protein folding**. In these folders except **A bistable gradient system with colored noise**,

 `code for saving matrics.py` is used to train RC, save matrics and draw deterministic part of the trained model. 
 
 `code for deterministic.py` is used to draw deterministic part of the trained model with saved matircs. 
 
 `code for evaluation.py` is used to draw the evaluation with saved matrics.

 While in **A bistable gradient system with colored noise**,

`code for learning noise.py` is used to learn the separated noise data by second RC.

`code Total for predicting phase.py` is used to generate average trajectory of the predictions with trained model and learned noise data.

`code for drawing.py` is used to draw the average trajectory and absolute errro of the results.

(2) Supplementary Information: code for the examples in the Supplementary Information.

(3) previous approaches: code for three types of previous approaches, including **SINDy** , **RNN** , and **filters**.

(4) fast Fourier transform: code for FFT and an example **A bistable gradient system with colored noise**.

(5) force learning and PSD: code for **force learning** and **PSD**.

## Data

When we trained the RC, some matrics would be saved in a path, and we would load them in the predicting phase. We divided the data into several folders, the folder with the same name as in the code means that the data here points to this folder.

## Example

We provide a demonstration example. 

## A Note on Results
For systems with white noise, 'np.random' functions of random matrices are necessary for both data generation and noise separation (def train1/2, def DoubleWell). This can lead to minor differences in 'NoiseToUse' and 'SampledNoise' with each run. Consequently, even though we provide the trained model, the results of rolling predictions may exhibit slight variations each time, as the scenario is not exactly the same as when these models were trained, but the overall effect is approximately similar. For the real data example, sometimes the 'np.random' may lead to numerical error. The result can be reproduced by running it several times, and we provide an ipynb file for demonstration (np.random.seed(42)).

For systems with colored noise (two Lorenz-63 and one Lorenz-96), we saved the "NoiseTemp" and provided the predicted noise "NoiseTotal.npy" of 50 times predictions, and the matrices of the deterministic RC. To generate transition trajectories, run the provided script code_Total_for_predicting_phase.py. The prediction for FIG. 3(d) may differ slightly from the version in the paper (green line), due to the data for this figure in the arXiv version being inadvertently overwritten by Supplementary FIG. 20 (represent the same system with different parameters). Therefore, we provide a correct version of the prediction.



