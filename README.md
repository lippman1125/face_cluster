# Face Cluster
we use chinese-whisper algorithm to perform face cluster function, and use RI/Precision/Recall to metric algorithm.
# installion
```
make all
```
# Test
```
face_cluster facedir_list.tx root_dir
```
# Result
we test on our private face dataset, result as follow:
  
|RI average|Precision average|Recall average|
|:---:|:---:|:---:| 
|0.959592| 0.96861| 0.880657|

# Explatation
![ri](https://github.com/lippman1125/github_images/blob/master/cluster_images/random_index.png)  
 