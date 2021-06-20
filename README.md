# iNeRF: Inverting Neural Radiance Fields for Pose Estimation 
Implementation of iNeRF project using PyTorch

## Example results

![](https://user-images.githubusercontent.com/63703454/122670629-4f0cd100-d1c3-11eb-9216-94f1bc7ef047.gif)
![](https://user-images.githubusercontent.com/63703454/122670771-f5f16d00-d1c3-11eb-82a7-6446f1f05a95.gif)
![](https://user-images.githubusercontent.com/63703454/122670773-f7229a00-d1c3-11eb-99be-621e4547a768.gif)


## Different sampling strategies 

![](https://user-images.githubusercontent.com/63703454/122685702-61abf800-d20d-11eb-8f26-39b4d9ae37c0.gif)
![](https://user-images.githubusercontent.com/63703454/122685701-61136180-d20d-11eb-9110-6d481ea2199c.gif)
![](https://user-images.githubusercontent.com/63703454/122685703-62448e80-d20d-11eb-9d80-9d07bbfdceb8.gif)

Left - **random**, in the middle - **interest points**, right - **interest regions**. 
Interest regions sampling strategy provides faster convergence and doesnt stuck in a local minimum like interest points. 
