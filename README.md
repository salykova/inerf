# iNeRF: Inverting Neural Radiance Fields for Pose Estimation 
Implementation of iNeRF project using PyTorch

## Example results

![](https://user-images.githubusercontent.com/63703454/122670771-f5f16d00-d1c3-11eb-82a7-6446f1f05a95.gif)
![](https://user-images.githubusercontent.com/63703454/122670773-f7229a00-d1c3-11eb-99be-621e4547a768.gif)


## Different sampling strategies 

![](https://user-images.githubusercontent.com/63703454/122686222-51e1e300-d210-11eb-8f4c-be25f078ffa9.gif)
![](https://user-images.githubusercontent.com/63703454/122686229-58705a80-d210-11eb-9c0f-d6c2208b5457.gif)
![](https://user-images.githubusercontent.com/63703454/122686235-5ad2b480-d210-11eb-87ec-d645ae07b8d7.gif)

Left - **random**, in the middle - **interest points**, right - **interest regions**. 
Interest regions sampling strategy provides faster convergence and doesnt stuck in a local minimum like interest points. 
