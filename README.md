# CoSimGNN:  Towards Large-scale Graph Similarity Computation
![CoSimGNN](./CoSimGNN.PNG)

## Requirements

To install requirements:

```setup
Using pip:
	pip install -r requirements_pip.txt

Using conda:
	conda create -n your_env_name --file requirements_conda.txt
```

## Training

To train the model in the paper, run this command:

```train
python main.py --dataset <dataset> --num_iters <number_of_training_iterations> --model <model_name> --dos_true 'dist'
```

\<dataset> can be among \{BA_60, BA_100, BA_200, ER_100, IMDB-L, Enzymes}.

\<model_name> can be among \{GCN-Mean, GCN-Max, simgnn, gsim_cnn, GMN, CoSim-CNN, CoSim-Mem, CoSim-ATT, CoSim-SAG, CoSim-TOPK, CoSim-GNN10, CoSim-GNN1}

## Evaluation

To evaluate the models, run:

```eval
python main.py --dataset <dataset> --num_iters <number_of_training_iterations> --model <model_name> --dos_true 'dist' --load_model <model.pth>
```

## Pre-trained Models

Download pretrained models here:

- [Pre-trained_models](https://github.com/KDD2021/code_for_CoSimGNN/tree/main/Models_in_CoSimGNN)

## Results

The results are shown as below:

<table>
   <tr>
      <td>model</td>
      <td>BA-60</td>
      <td></td>
      <td>BA-100</td>
      <td></td>
      <td>BA-200</td>
      <td></td>
      <td>ER-100</td>
      <td></td>
      <td>IMDB-L</td>
      <td></td>
      <td>Enzymes</td>
      <td></td>
   </tr>
   <tr>
      <td></td>
      <td>MSE</td>
      <td>MAE</td>
      <td>MSE</td>
      <td>MAE</td>
      <td>MSE</td>
      <td>MAE</td>
      <td>MSE</td>
      <td>MAE</td>
      <td>MSE</td>
      <td>MAE</td>
      <td>MSE</td>
      <td>MAE</td>
   </tr>
   <tr>
      <td>GCN-MEAN</td>
      <td>5.85</td>
      <td>53.92</td>
      <td>12.53</td>
      <td>90.88</td>
      <td>23.66</td>
      <td>127.58</td>
      <td>16.58</td>
      <td>92.98</td>
      <td>22.17</td>
      <td>55.35</td>
      <td>10.69</td>
      <td>61.09</td>
   </tr>
   <tr>
      <td>GCN-MAX</td>
      <td>13.66</td>
      <td>91.38</td>
      <td>12.04</td>
      <td>85.44</td>
      <td>22.77</td>
      <td>107.58</td>
      <td>79.09</td>
      <td>211.07</td>
      <td>47.14</td>
      <td>123.16</td>
      <td>13.15</td>
      <td>67.64</td>
   </tr>
   <tr>
      <td>SIMGNN</td>
      <td>8.57</td>
      <td>63.8</td>
      <td>6.23</td>
      <td>47.8</td>
      <td>3.06</td>
      <td>32.77</td>
      <td>6.37</td>
      <td>45.3</td>
      <td>7.42</td>
      <td>33.74</td>
      <td>13.96</td>
      <td>49.04</td>
   </tr>
   <tr>
      <td>GSIMCNN</td>
      <td>5.97</td>
      <td>56.05</td>
      <td>1.86</td>
      <td>30.18</td>
      <td>2.35</td>
      <td>32.64</td>
      <td>2.93</td>
      <td>34.41</td>
      <td>5.01</td>
      <td>30.43</td>
      <td>2.48</td>
      <td>26.79</td>
   </tr>
   <tr>
      <td>GMN</td>
      <td>2.82</td>
      <td>38.38</td>
      <td>4.14</td>
      <td>34.17</td>
      <td>1.16</td>
      <td>26.6</td>
      <td>1.59</td>
      <td>28.68</td>
      <td>3.82</td>
      <td>27.28</td>
      <td>3.02</td>
      <td>35.86</td>
   </tr>
   <tr>
      <td>COSIM-CNN</td>
      <td>2.5</td>
      <td>35.53</td>
      <td>1.49</td>
      <td>27.2</td>
      <td>0.53</td>
      <td>18.44</td>
      <td>2.78</td>
      <td>33.36</td>
      <td>10.37</td>
      <td>38.03</td>
      <td>26.64</td>
      <td>104.05</td>
   </tr>
   <tr>
      <td>COSIM-ATT</td>
      <td>2.04</td>
      <td>33.41</td>
      <td>0.97</td>
      <td>22.95</td>
      <td>0.73</td>
      <td>16.26</td>
      <td>1.39</td>
      <td>27.27</td>
      <td>1.53</td>
      <td>16.57</td>
      <td>1.48</td>
      <td>25.95</td>
   </tr>
   <tr>
      <td>COSIM-SAG</td>
      <td>3.26</td>
      <td>38.85</td>
      <td>3.3</td>
      <td>33.14</td>
      <td>1.91</td>
      <td>35.48</td>
      <td>1.55</td>
      <td>29.84</td>
      <td>1.62</td>
      <td>16.08</td>
      <td>1.33</td>
      <td>28.19</td>
   </tr>
   <tr>
      <td>COSIM-TOPK</td>
      <td>3.44</td>
      <td>40.87</td>
      <td>1.24</td>
      <td>25.61</td>
      <td>0.88</td>
      <td>20.63</td>
      <td>2.04</td>
      <td>34.28</td>
      <td>1.98</td>
      <td>20.02</td>
      <td>1.99</td>
      <td>28.95</td>
   </tr>
   <tr>
      <td>COSIM-MEM</td>
      <td>5.45</td>
      <td>48.07</td>
      <td>1.11</td>
      <td>24.59</td>
      <td>0.32</td>
      <td>14.82</td>
      <td>1.74</td>
      <td>26.78</td>
      <td>1.57</td>
      <td>17.02</td>
      <td>1.21</td>
      <td>26.22</td>
   </tr>
   <tr>
      <td>COSIM-GNN10</td>
      <td>2.04</td>
      <td>33.04</td>
      <td>1.01</td>
      <td>23.53</td>
      <td>0.40</td>
      <td>16.43</td>
      <td>1.38</td>
      <td>27.43</td>
      <td>1.68</td>
      <td>17.57</td>
      <td>1.38</td>
      <td>24.73</td>
   </tr>
   <tr>
      <td>COSIM-GNN1</td>
      <td>1.84</td>
      <td>32.36</td>
      <td>0.95</td>
      <td>22.06</td>
      <td>0.36</td>
      <td>15.42</td>
      <td>1.17</td>
      <td>25.73</td>
      <td>2.00</td>
      <td>18.62</td>
      <td>1.09</td>
      <td>25.88</td>
   </tr>
</table>


To reproduce the results, just download the trained models and set <model.pth> below to any trained models trained under specific dataset. The code will automatically extract all the trained models and validate them on the validation set to choose the best one on the validation set for testing.

```eval
python main.py --dataset <dataset> --num_iters <number_of_training_iterations> --model <model_name> --dos_true 'dist' --load_model <model.pth>
```


## Contributing

> 📋Pick a licence and describe how to contribute to your code repository. 