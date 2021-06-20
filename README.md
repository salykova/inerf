# iNeRF: Inverting Neural Radiance Fields for Pose Estimation 
Implementation of iNeRF project using PyTorch

## Example results

![](https://user-images.githubusercontent.com/63703454/122670771-f5f16d00-d1c3-11eb-82a7-6446f1f05a95.gif)
![](https://user-images.githubusercontent.com/63703454/122670773-f7229a00-d1c3-11eb-99be-621e4547a768.gif)


## Different sampling strategies 

![](https://user-images.githubusercontent.com/63703454/122686046-843f1080-d20f-11eb-975e-18ff257ccb64.gif)
![](https://user-images.githubusercontent.com/63703454/122686095-c2d4cb00-d20f-11eb-9502-47a52c7e6a8d.gif)
![](https://user-images.githubusercontent.com/63703454/122686099-c5cfbb80-d20f-11eb-89b0-2f91182a9b08.gif)

Left - **random**, in the middle - **interest points**, right - **interest regions**. 
Interest regions sampling strategy provides faster convergence and doesnt stuck in a local minimum like interest points. 
