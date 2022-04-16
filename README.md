# iNeRF: Inverting Neural Radiance Fields for 6-DoF Pose Estimation 
Implementation of [Inverting Neural Radiance Fields for Pose Estimation](https://arxiv.org/abs/2012.05877) using PyTorch.
## Installation
To start, I recommend to create an environment using conda:
```
conda create -n inerf python=3.8
conda activate inerf
```
Clone the repository and install dependencies:
```
git clone https://github.com/salykovaa/inerf.git
cd inerf
pip install -r requirements.txt
```
## How to use
To run the algorithm on _Lego_ object
```
python run.py --config configs/lego.txt
```
If you want to store gif video of optimization process, set ```OVERLAY = True``` [here](https://github.com/salykovaa/inerf/blob/a8c996958789168b93e73ed8aee8d6f76ceb0fbc/run.py#L217)

All other parameters such as _batch size_, _sampling strategy_, _initial camera error_ you can adjust in corresponding config [files](https://github.com/salykovaa/inerf/tree/main/configs).

To run the algorithm on the llff dataset, just download the "nerf_llff_data" folder from [here](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) and put the downloaded folder in the "data" folder.

All NeRF models were trained using this code [https://github.com/yenchenlin/nerf-pytorch/](https://github.com/yenchenlin/nerf-pytorch/)
```
├── data 
│   ├── nerf_llff_data   
│   ├── nerf_synthetic  
```
## Examples

![](https://user-images.githubusercontent.com/63703454/125823439-4d89d5fa-4aa6-4159-9df4-4fcf55441632.gif)
![](https://user-images.githubusercontent.com/63703454/126023953-682a28fe-2bb7-419a-98f8-da3a139a81bf.gif)

![](https://user-images.githubusercontent.com/63703454/122670771-f5f16d00-d1c3-11eb-82a7-6446f1f05a95.gif)
![](https://user-images.githubusercontent.com/63703454/122670773-f7229a00-d1c3-11eb-99be-621e4547a768.gif)

## Different sampling strategies 

![](https://user-images.githubusercontent.com/63703454/122686222-51e1e300-d210-11eb-8f4c-be25f078ffa9.gif)
![](https://user-images.githubusercontent.com/63703454/122686229-58705a80-d210-11eb-9c0f-d6c2208b5457.gif)
![](https://user-images.githubusercontent.com/63703454/122686235-5ad2b480-d210-11eb-87ec-d645ae07b8d7.gif)

Left - **random**, in the middle - **interest points**, right - **interest regions**. 
Interest regions sampling strategy provides faster convergence and doesnt stick in a local minimum like interest points. 

## Citation
Kudos to the authors
```
@article{yen2020inerf,
  title={{iNeRF}: Inverting Neural Radiance Fields for Pose Estimation},
  author={Lin Yen-Chen and Pete Florence and Jonathan T. Barron and Alberto Rodriguez and Phillip Isola and Tsung-Yi Lin},
  year={2020},
  journal={arxiv arXiv:2012.05877},
}
```
Parts of the code were based on yenchenlin's NeRF implementation: https://github.com/yenchenlin/nerf-pytorch
```
@misc{lin2020nerfpytorch,
  title={NeRF-pytorch},
  author={Yen-Chen, Lin},
  howpublished={\url{https://github.com/yenchenlin/nerf-pytorch/}},
  year={2020}
}
```
