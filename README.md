# BaiduCityAttr

Estimating gravitational attraction of Chinese cities using particle swarm optimization, based on Baidu search index.  
See our paper for more details:  
Guo, H., Zhang, W., Du, H., Kang, C., Liu, Y. (2022). Understanding China’s urban system evolution from web search index data. EPJ Data Science, 11, 20.

## Data
We provide annually averaged Baidu search index between cities. The data cover 322 cities for years 2011 to 2016, and 357 cities for 2017 to 2019. Note that the data is smoothed to remove the influence of public emergencies (see our paper for detail). The indices 0-321 (0-356) in the text data correspond to "city_list" in the code "cpso_bbj_cuda.cpp".

## Code
We provide CUDA code to reversely fit the directed gravity model. The method is CPSO-H + BBJ, which integrates two variants of particle swarm optimization. GPU-based parallel computing is supported.  
The final code is cpso_bbj_cuda.cpp, other versions are kept for references of my B.S. thesis.  
If you are interested in optimization of reverse gravity models, feel free to contact me via e-mail.  
