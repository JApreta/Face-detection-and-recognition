Conv1=(64 * 3 * 3 * 3) + 64
Conv2=(128 * 3 * 3 * 64) + 128
MaxPool2= 0
Conv3= (256 * 3 * 3 * 128) + 256
Conv4= (512 * 3 * 3 * 256) + 512
MaxPool4= 0
Conv5= (512 * 3 * 3 * 512) + 512
MaxPool5= 0
FC1= (7 * 7 * 512 * 4096) + 4096
FC2= (4096 * 4096) + 4096
FC3= (4096 * 1000) + 1000


print(Conv1+ Conv2+Conv3+Conv4+Conv5+FC1+FC2+FC3)
